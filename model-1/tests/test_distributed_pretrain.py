import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from aim.contracts import ContractError
from aim.corpus import Corpus, intake
from aim.corpus_fixture import create_fixture
from aim.sharded_data import RankCorpus, partition, audit_partition
from aim.tokenization import ByteTokenizer, fit_bpe
from aim.token_stream import TokenStream
from aim.tracking import ROOT, digest

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.corpus = Corpus(intake(create_fixture(self.root/"fixture"), self.root/"intake"))
        self.config = json.loads((ROOT/"configs/distributed-pretrain-smoke.json").read_text())


class PartitionTests(Fixture):
    def test_partition_coverage_and_cursor_rank_binding(self):
        for world in (1, 2, 4, 12):
            layout = partition(self.corpus, world)
            self.assertTrue(audit_partition(layout, self.corpus)["disjoint_complete_ownership"])
            self.assertEqual(len({i for o in layout["owners"] for i in o["ids"]}), 12)
        stream = TokenStream(RankCorpus(self.corpus, 0, 2), ByteTokenizer())
        stream.batch(2, 16)
        with self.assertRaises(ContractError):
            TokenStream(RankCorpus(self.corpus, 1, 2), ByteTokenizer(), cursor=stream.cursor())
        with self.assertRaises(ContractError):
            TokenStream(RankCorpus(self.corpus, 0, 4), ByteTokenizer(), cursor=stream.cursor())

    def test_empty_rank_invalid_world_and_cross_rank_read(self):
        for value in (0, 13, True, 1.5):
            with self.assertRaises(ContractError):
                partition(self.corpus, value)
        shard = RankCorpus(self.corpus, 0, 2)
        other = RankCorpus(self.corpus, 1, 2).records("train")[0]
        with self.assertRaises(ContractError):
            shard.text(other)
        with self.assertRaises(ContractError):
            shard.records("validation")

    def test_partition_tampering(self):
        layout = partition(self.corpus, 2)
        layout["owners"][0]["ids"].append(layout["owners"][1]["ids"][0])
        with self.assertRaises(ContractError):
            audit_partition(layout, self.corpus)


@unittest.skipUnless(HAS_TORCH, "Pinned torch environment unavailable")
class CheckpointTests(Fixture):
    def staged(self):
        import torch
        from dataclasses import asdict
        from aim.neural import CausalLM, ModelConfig
        from aim.checkpoint_bundle import SCHEMA, prepare, save_record, tree_hash
        from aim.training import seed_all
        seed_all(17)
        cfg = ModelConfig(width=8, layers=1, heads=2, kv_heads=1, ffn_width=16, context=8)
        model = CausalLM(cfg)
        optimizer = torch.optim.AdamW(model.parameters())
        tokenizer = ByteTokenizer()
        layout = partition(self.corpus, 2)
        common = {"schema": SCHEMA, "kind": "distributed_pretraining_lm", "origin": "aim-random-init-v1", "world_size": 2,
                  "step": 1, "model_config": asdict(cfg), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                  "stable_config": {}, "config_hash": digest({}), "tokenizer": tokenizer.specification(),
                  "tokenizer_hash": digest(tokenizer.specification()), "corpus_hash": self.corpus.fingerprint,
                  "partition_hash": layout["partition_hash"], "global_tokens": 16}
        pending = prepare(self.root/"checkpoints", 1)
        states = []
        for rank in range(2):
            stream = TokenStream(RankCorpus(self.corpus, rank, 2), tokenizer)
            stream.batch(1, 8)
            state = {k: common[k] for k in ("schema", "origin", "world_size", "step", "config_hash", "tokenizer_hash", "corpus_hash", "partition_hash", "global_tokens")}
            state.update(kind="distributed_rank", rank=rank, local_tokens=8, cursor=stream.cursor(), torch_rng=torch.get_rng_state(),
                         model_hash=tree_hash(model.state_dict()), optimizer_hash=tree_hash(optimizer.state_dict()))
            save_record(pending/f"rank-{rank:03d}.pt", state)
            states.append(state)
        return pending, common, states

    def test_publication_partial_rejection_and_roundtrip(self):
        from aim.checkpoint_bundle import load_bundle, publish, scan
        from aim.distributed_jobs import audit_checkpoint
        pending, common, _ = self.staged()
        with self.assertRaises(ContractError):
            load_bundle(pending)
        final = publish(pending, common)
        self.assertFalse(pending.exists())
        self.assertEqual(load_bundle(final)[0]["step"], 1)
        self.assertTrue(audit_checkpoint(final, self.corpus.path)["cursor_bindings_valid"])
        self.assertEqual(scan(final.parent)[0]["status"], "VALID")

    def test_incomplete_or_divergent_rank_cannot_publish(self):
        from aim.checkpoint_bundle import publish, validate_states
        pending, common, states = self.staged()
        bad = copy.deepcopy(states)
        bad[1]["model_hash"] = "wrong"
        with self.assertRaises(ContractError):
            validate_states(common, bad)
        (pending/"rank-001.pt").unlink()
        with self.assertRaises(ContractError):
            publish(pending, common)
        self.assertFalse((pending.parent/"step-000001").exists())

    def test_corruption_and_manifest_path_injection(self):
        from aim.checkpoint_bundle import publish, load_bundle, scan
        pending, common, _ = self.staged()
        final = publish(pending, common)
        (final/"rank-000.pt").write_bytes(b"corruption")
        with self.assertRaises(ContractError):
            load_bundle(final)
        self.assertEqual(scan(final.parent)[0]["status"], "INVALID")
        manifest = json.loads((final/"bundle.json").read_text())
        manifest["files"] = {"../outside.pt": "0"*64}
        manifest["bundle_hash"] = digest({k: v for k,v in manifest.items() if k != "bundle_hash"})
        (final/"bundle.json").write_text(json.dumps(manifest))
        with self.assertRaises(ContractError):
            load_bundle(final)

    def test_global_token_gradient_not_mean_of_rank_means(self):
        import torch
        from aim.distributed_pretrain import scaled_loss
        w = torch.tensor(.4, requires_grad=True)
        global_loss = ((w-1)**2+3*(w+1)**2)/4
        expected = torch.autograd.grad(global_loss, w)[0]
        actual = torch.autograd.grad((scaled_loss((w-1)**2, 2, 4)+scaled_loss(3*(w+1)**2, 2, 4))/2, w)[0]
        naive = torch.autograd.grad(((w-1)**2+(w+1)**2)/2, w)[0]
        self.assertTrue(torch.equal(expected, actual))
        self.assertNotEqual(float(expected), float(naive))

    def test_preflight_and_config_guards(self):
        from aim.distributed_jobs import preflight
        report = preflight(self.config, self.corpus.path, world_size=2)
        self.assertEqual(report["effective_sequences_per_update"], 8)
        for key, value in (("accumulation_steps", 0), ("collective_timeout_seconds", 1000)):
            with self.assertRaises(ContractError):
                preflight({**self.config, key: value}, self.corpus.path)


@unittest.skipUnless(HAS_TORCH, "Pinned torch environment unavailable")
class DistributedIntegrationTests(Fixture):
    def test_two_rank_serial_oracle_resume_and_sft_export(self):
        from aim.distributed_jobs import launch, audit_job, export_initialization
        from aim.distributed_reference import serial_reference, compare_serial, compare_resume
        from aim.training import train
        config = {**self.config, "steps": 4}
        full = launch(config, self.corpus.path, self.root/"jobs")
        half = launch({**config, "steps": 2}, self.corpus.path, self.root/"jobs")
        resumed = launch(config, self.corpus.path, self.root/"jobs", resume=half/"job/checkpoints/step-000002")
        final = full/"job/checkpoints/step-000004"
        self.assertTrue(compare_resume(final, resumed/"job/checkpoints/step-000004")["exact"])
        reference = serial_reference(config, self.corpus.path, self.root/"serial")
        self.assertLessEqual(compare_serial(reference, final)["max_absolute_parameter_difference"], 2e-6)
        self.assertEqual(audit_job(full, self.corpus.path)["replayed_rank_updates"], 8)
        metrics_files = sorted((full/"job/ranks").glob("*/metrics.json"))
        originals = [p.read_text() for p in metrics_files]
        try:
            for i, p in enumerate(metrics_files):
                value = json.loads(originals[i])
                value["local_tokens_total"] += 1 if i == 0 else -1
                p.write_text(json.dumps(value))
            with self.assertRaises(ContractError):
                audit_job(full, self.corpus.path)
        finally:
            for p, text in zip(metrics_files, originals):
                p.write_text(text)
        exported = export_initialization(final, self.root/"exports")
        sft = {k: v for k, v in config.items() if k not in {"checkpoint_every", "eval_batches", "accumulation_steps", "collective_timeout_seconds"}}
        sft.update(stage="sft", steps=1)
        self.assertTrue((train(sft, self.root/"sft", initialize=exported)/"checkpoint.pt").is_file())

    def test_worker_failure_keeps_prior_checkpoint_and_rejects_pending(self):
        from aim.distributed_jobs import launch, DistributedJobError
        from aim.checkpoint_bundle import scan, load_bundle
        config = {**self.config, "steps": 4}
        with self.assertRaises(DistributedJobError) as caught:
            launch(config, self.corpus.path, self.root/"jobs", fault={"rank": 1, "step": 4, "mode": "before_publish"})
        path = caught.exception.path
        self.assertEqual(json.loads((path/"status.json").read_text())["status"], "FAILED")
        self.assertIn("INJECTED_WORKER_FAILURE", (path/"launcher.log").read_text())
        records = scan(path/"job/checkpoints")
        self.assertEqual({r["status"] for r in records}, {"INCOMPLETE", "VALID"})
        with self.assertRaises(ContractError):
            load_bundle(path/"job/checkpoints/.pending-step-000004")
        recovered = launch(config, self.corpus.path, self.root/"jobs", resume=path/"job/checkpoints/step-000002")
        self.assertTrue((recovered/"data-audit.json").is_file())

    def test_bpe_workers_and_world_size_rejection(self):
        from aim.distributed_jobs import launch
        from aim.distributed_pretrain import resume_state
        from aim.checkpoint_bundle import load_bundle
        tokenizer = fit_bpe(self.corpus, 8)
        path = launch({**self.config, "steps": 2}, self.corpus.path, self.root/"jobs", tokenizer=tokenizer)
        common, states, _ = load_bundle(path/"job/checkpoints/step-000002")
        with self.assertRaisesRegex(ContractError, "world-size"):
            resume_state(common, states, self.config, self.corpus, tokenizer, partition(self.corpus, 4), 0)


if __name__ == "__main__":
    unittest.main()

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aim.contracts import ContractError
from aim.corpus import Corpus, intake, MAX_DOCUMENT_BYTES
from aim.corpus_fixture import create_fixture
from aim.tokenization import ByteTokenizer, fit_bpe, tokenizer_from_spec, compare_tokenizers
from aim.token_stream import TokenStream
from aim.tracking import digest

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None


class CorpusFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = create_fixture(self.root/"input")

    def mutate(self, fn):
        manifest = json.loads(self.manifest.read_text())
        fn(manifest)
        self.manifest.write_text(json.dumps(manifest))

    def corpus(self):
        return Corpus(intake(self.manifest, self.root/"runs"))


class CorpusTests(CorpusFixture):
    def test_intake_is_immutable_and_metadata_bound(self):
        corpus = self.corpus()
        self.assertEqual(len(corpus.index["documents"]), 20)
        self.assertEqual(len(corpus.records("train")), 12)
        row = corpus.records("train")[0]
        original = corpus.text(row)
        (self.manifest.parent/row["path"]).write_text("changed original")
        self.assertEqual(corpus.text(row), original)
        self.assertEqual(corpus.fingerprint, Corpus(corpus.path).fingerprint)

    def test_rights_and_privacy_are_required(self):
        for field, key, value in (("rights", "training_allowed", False), ("rights", "training_allowed", "true"), ("privacy_review", "status", "pending")):
            original = self.manifest.read_text()
            self.mutate(lambda m: m["documents"][0][field].update({key: value}))
            with self.assertRaises(ContractError):
                self.corpus()
            self.manifest.write_text(original)
        failed = list((self.root/"runs").glob("*/status.json"))
        self.assertEqual(len(failed), 3)
        self.assertTrue(all(json.loads(p.read_text())["status"] == "FAILED" for p in failed))

    def test_hash_and_size_mismatch_rejected(self):
        row = json.loads(self.manifest.read_text())["documents"][0]
        path = self.manifest.parent/row["path"]
        path.write_text("unexpected change")
        with self.assertRaises(ContractError):
            self.corpus()
        path.write_text("x"*(MAX_DOCUMENT_BYTES+1))
        with self.assertRaises(ContractError):
            self.corpus()

    def test_family_split_leakage_rejected(self):
        self.mutate(lambda m: m["documents"][12].update(group=m["documents"][0]["group"]))
        with self.assertRaisesRegex(ContractError, "family leaks"):
            self.corpus()

    def test_exact_and_normalized_duplicates_rejected(self):
        for whitespace in (False, True):
            original = self.manifest.read_text()
            data = json.loads(original)
            left, right = data["documents"][0], data["documents"][12]
            text = (self.manifest.parent/left["path"]).read_text()
            text = "  "+text.replace(" ", "   ")+" \n" if whitespace else text
            (self.manifest.parent/right["path"]).write_text(text)
            right["sha256"] = digest(text.encode())
            self.manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ContractError, "duplicate"):
                self.corpus()
            self.manifest.write_text(original)

    def test_paths_symlinks_and_binary_rejected(self):
        original = self.manifest.read_text()
        for bad in ("../outside.txt", str(self.manifest)):
            self.mutate(lambda m: m["documents"][0].update(path=bad))
            with self.assertRaises(ContractError):
                self.corpus()
            self.manifest.write_text(original)
        row = json.loads(original)["documents"][0]
        path = self.manifest.parent/row["path"]
        link = self.manifest.parent/"linked.txt"
        link.symlink_to(path)
        self.mutate(lambda m: m["documents"][0].update(path="linked.txt"))
        with self.assertRaises(ContractError):
            self.corpus()
        self.manifest.write_text(original)
        path.write_bytes(b"\xff\x00")
        self.mutate(lambda m: m["documents"][0].update(sha256=digest(b"\xff\x00")))
        with self.assertRaises(ContractError):
            self.corpus()

    def test_corpus_object_and_index_tampering(self):
        corpus = self.corpus()
        row = corpus.records("train")[0]
        obj = corpus.path.parent/"objects"/row["sha256"]
        obj.write_bytes(b"x"*row["bytes"])
        with self.assertRaises(ContractError):
            corpus.text(row)
        doc = json.loads(corpus.path.read_text())
        doc["documents"][0]["split"] = "test"
        corpus.path.write_text(json.dumps(doc))
        with self.assertRaises(ContractError):
            Corpus(corpus.path)


class TokenizerStreamTests(CorpusFixture):
    def test_tokenizer_roundtrip_determinism_and_fit_partition(self):
        corpus = self.corpus()
        original = corpus.text
        reads = []
        def tracked(row):
            reads.append(row["split"])
            return original(row)
        with patch.object(corpus, "text", side_effect=tracked):
            tokenizer = fit_bpe(corpus, 32)
        self.assertEqual(set(reads), {"train"})
        self.assertEqual(tokenizer.specification(), fit_bpe(corpus, 32).specification())
        for text in ("", "α हिन्दी 数学 café 🧪", "\x00\n\t", "aaaaa", "unseen 🚀 emoji"):
            for candidate in (ByteTokenizer(), tokenizer, tokenizer_from_spec(tokenizer.specification())):
                self.assertEqual(candidate.decode(candidate.encode(text)), text)
        comparison = compare_tokenizers(corpus, {"byte": ByteTokenizer(), "bpe": tokenizer})
        self.assertEqual(len(comparison["bpe"]["documents"]), 4)
        with self.assertRaises(ContractError):
            compare_tokenizers(corpus, {"byte": ByteTokenizer()}, "test")

    def test_invalid_tokenizer_spec(self):
        spec = fit_bpe(self.corpus(), 2).specification()
        for invalid in ({**spec, "vocab_size": 1}, {**spec, "merges": [[256, 1]]}, {**spec, "merges": [[99999, 0]]}):
            with self.assertRaises(ContractError):
                tokenizer_from_spec(invalid)

    def test_stream_pairs_boundaries_padding_and_no_dropped_tokens(self):
        corpus, tokenizer = self.corpus(), ByteTokenizer()
        tokens = [token for _, text in corpus.documents("validation") for token in [256]+tokenizer.encode(text)+[257]]
        stream = TokenStream(corpus, tokenizer, "validation", repeat=False)
        actual, scored = [], []
        while True:
            try:
                batch = stream.batch(3, 17)
            except StopIteration:
                break
            for x, y, mask in zip(batch["inputs"], batch["targets"], batch["mask"]):
                actual.extend((a, b) for a, b in zip(x, y) if b != 258)
                scored.extend(b for b, m in zip(y, mask) if m)
        self.assertEqual(actual, list(zip(tokens, tokens[1:])))
        self.assertEqual(scored, [t for t in tokens[1:] if t != 256])
        restored = TokenStream(corpus, tokenizer, "validation", False, stream.cursor())
        with self.assertRaises(StopIteration):
            restored.batch(1, 16)

    def test_cursor_resume_across_epochs_and_tokenizers(self):
        corpus = self.corpus()
        for tokenizer in (ByteTokenizer(), fit_bpe(corpus, 8)):
            stream = TokenStream(corpus, tokenizer)
            for _ in range(5):
                stream.batch(16, 64)
            self.assertGreater(stream.epoch, 0)
            resumed = TokenStream(corpus, tokenizer, cursor=stream.cursor())
            for _ in range(4):
                self.assertEqual(stream.batch(3, 31), resumed.batch(3, 31))
            bad = stream.cursor(); bad["carry"][0] += 1
            with self.assertRaises(ContractError):
                TokenStream(corpus, tokenizer, cursor=bad)
            with self.assertRaises(ContractError):
                TokenStream(corpus, tokenizer, "validation", cursor=stream.cursor())


@unittest.skipUnless(HAS_TORCH, "Pinned torch environment unavailable")
class PretrainingTests(CorpusFixture):
    def config(self, steps=4):
        return {"seed": 17, "steps": steps, "batch_size": 2, "learning_rate": .001, "weight_decay": .01,
                "checkpoint_every": 2, "eval_batches": 2,
                "model": {"width": 16, "layers": 1, "heads": 2, "kv_heads": 1, "ffn_width": 32, "context": 64}}

    def test_pretraining_resume_and_real_failure_recovery(self):
        import torch
        from aim.pretrain import pretrain, load_pretrained
        from aim.neural import read_checkpoint
        from aim.training import save_checkpoint
        corpus = self.corpus()
        full = pretrain(self.config(), corpus.path, self.root/"training")
        def fail_after_saved(run, record, filename):
            save_checkpoint(run, record, filename)
            raise RuntimeError("injected interruption after a durable checkpoint")
        with patch("aim.training.save_checkpoint", side_effect=fail_after_saved):
            with self.assertRaisesRegex(RuntimeError, "injected interruption"):
                pretrain(self.config(), corpus.path, self.root/"failed")
        failed = next((self.root/"failed").iterdir())
        self.assertEqual(json.loads((failed/"status.json").read_text())["status"], "FAILED")
        recovered = pretrain(self.config(), corpus.path, self.root/"training", resume=failed/"checkpoint-step-000002.pt")
        a, b = (read_checkpoint(p/"checkpoint.pt") for p in (full, recovered))
        for k in a["model"]:
            self.assertTrue(torch.equal(a["model"][k], b["model"][k]))
        self.assertEqual(a["cursor"], b["cursor"])
        self.assertEqual(a["scored_tokens"], b["scored_tokens"])
        self.assertTrue(torch.equal(a["torch_rng"], b["torch_rng"]))
        ma, mb = (json.loads((p/"metrics.json").read_text()) for p in (full, recovered))
        self.assertEqual(ma["final"], mb["final"])
        self.assertEqual([u["batch_hash"] for u in ma["updates"]][2:], [u["batch_hash"] for u in mb["updates"]])
        load_pretrained(full/"checkpoint.pt")
        with self.assertRaises(ContractError):
            pretrain({**self.config(6), "learning_rate": .1}, corpus.path, self.root/"training", resume=full/"checkpoint.pt")

    def test_bpe_pretraining_resume_no_test_access_and_sft_bridge(self):
        import torch
        from aim.pretrain import pretrain
        from aim.neural import read_checkpoint
        from aim.training import train
        corpus = self.corpus()
        tokenizer = fit_bpe(corpus, 8)
        original = Corpus.text
        def no_test(instance, row):
            if row["split"] == "test":
                raise AssertionError("trainer read test text")
            return original(instance, row)
        with patch.object(Corpus, "text", no_test):
            full = pretrain(self.config(), corpus.path, self.root/"training", tokenizer)
            half = pretrain(self.config(2), corpus.path, self.root/"training", tokenizer)
            resumed = pretrain(self.config(), corpus.path, self.root/"training", tokenizer, half/"checkpoint.pt")
        a, b = (read_checkpoint(p/"checkpoint.pt") for p in (full, resumed))
        self.assertTrue(all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"]))
        sft = {k: v for k, v in self.config(1).items() if k not in {"checkpoint_every", "eval_batches"}}
        sft["stage"] = "sft"
        with self.assertRaises(ContractError):
            train(sft, self.root/"training", initialize=full/"checkpoint.pt")
        byte = pretrain(self.config(2), corpus.path, self.root/"training")
        fine = train(sft, self.root/"training", initialize=byte/"checkpoint.pt")
        self.assertEqual(read_checkpoint(fine/"checkpoint.pt")["stage"], "sft")
        with self.assertRaises(ContractError):
            pretrain(self.config(4), corpus.path, self.root/"training", tokenizer, byte/"checkpoint.pt")


if __name__ == "__main__":
    unittest.main()

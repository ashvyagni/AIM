import copy
import importlib.util
import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from aim.contracts import ContractError
from aim.datasets import arithmetic_data, arithmetic_reward, assert_disjoint, judge_data, load_supervised_dataset

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None
if HAS_TORCH:
    import torch
    from aim.neural import ByteTokenizer, CausalLM, DecisionNetwork, ModelConfig, read_checkpoint
    from aim.objectives import dpo_loss, reinforce_loss, proper_loss
    from aim.training import seed_all, sequence_scores, train


class DatasetTests(unittest.TestCase):
    def test_arithmetic_split_is_disjoint(self):
        assert_disjoint(*(arithmetic_data(s, 32) for s in ("train", "validation", "test")))

    def test_judge_group_splits_are_disjoint(self):
        sets = [judge_data(s, 32) for s in ("train", "validation", "test")]
        assert_disjoint(*sets, judge_data("test", 32, ood=True))

    def test_verifier_does_not_use_preference_label(self):
        row = {"operands":[2,3], "chosen":"999"}
        self.assertEqual(arithmetic_reward(row,"5"), 1)
        self.assertEqual(arithmetic_reward(row,"999"), 0)
        self.assertEqual(arithmetic_reward(row,"nonsense"), 0)

    def test_supervised_intake_and_label_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"dataset.json"
            row = {"id":"train-1", "group":"a", "prompt":"a", "response":"b", "chosen":"b", "rejected":"c",
                   "rights":"test-only", "label_origin":"synthetic"}
            record = {"schema":"aim-supervised-v1", "version":"test-1", "train":[row],
                      "validation":[{**row,"id":"val-1","group":"d","prompt":"d"}]}
            path.write_text(json.dumps(record))
            self.assertEqual(load_supervised_dataset(path)[2], "test-1")
            record["train"][0]["label_origin"] = "human"
            path.write_text(json.dumps(record))
            with self.assertRaises(ContractError):
                load_supervised_dataset(path)


@unittest.skipUnless(HAS_TORCH, "Optional pinned training environment is not installed")
class NeuralTests(unittest.TestCase):
    def setUp(self):
        seed_all(17)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.cfg = ModelConfig(width=16,layers=1,heads=2,kv_heads=1,ffn_width=32,context=64)

    def config(self, steps=2, stage="sft"):
        return {"stage":stage,"steps":steps,"batch_size":2,"seed":17,"learning_rate":0.001,
                "weight_decay":0.01,"model":asdict(self.cfg)}

    def test_unicode_tokenizer_roundtrip(self):
        tokenizer = ByteTokenizer()
        text = "AIM αβ 数学 🌍"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_parameter_accounting(self):
        model = CausalLM(self.cfg)
        self.assertEqual(sum(p.numel() for p in model.parameters()), self.cfg.parameter_estimate())
        self.assertEqual(sum(p.numel() for p in DecisionNetwork().parameters()), 225)

    def test_causal_prefix_invariance(self):
        model = CausalLM(self.cfg).eval()
        a = torch.tensor([[1,2,3,4]])
        b = torch.tensor([[1,2,50,60]])
        self.assertTrue(torch.allclose(model(a)[:,:2], model(b)[:,:2], atol=1e-7, rtol=0))

    def test_context_overflow_fails(self):
        with self.assertRaises(ContractError):
            CausalLM(self.cfg)(torch.ones((1,65),dtype=torch.long))

    def test_response_mask_and_padding(self):
        model, tokenizer = CausalLM(self.cfg), ByteTokenizer()
        pairs = [("abc=","xy"),("a=","z")]
        together, counts = sequence_scores(model, tokenizer, pairs)
        self.assertEqual(counts.tolist(), [3,2])
        for i, pair in enumerate(pairs):
            single, _ = sequence_scores(model,tokenizer,[pair])
            self.assertTrue(torch.allclose(single[0],together[i],atol=1e-5))
        with self.assertRaises(ContractError):
            sequence_scores(model,tokenizer,[("a"*100,"x")])

    def test_dpo_value_and_gradient(self):
        chosen = torch.tensor([0.0], requires_grad=True)
        rejected = torch.tensor([0.0], requires_grad=True)
        loss = dpo_loss(chosen,rejected,torch.zeros(1),torch.zeros(1),0.1)
        self.assertAlmostEqual(loss.item(),math.log(2),places=6)
        loss.backward()
        self.assertLess(chosen.grad.item(),0)
        self.assertGreater(rejected.grad.item(),0)

    def test_rlvr_reward_gradient_and_reference_frozen(self):
        logits = torch.zeros(3,requires_grad=True)
        reference = torch.zeros(3,requires_grad=True)
        loss, kl = reinforce_loss(logits,reference,torch.tensor([0.,1.,0.]),1,0.02)
        self.assertAlmostEqual(kl.item(),0)
        loss.backward()
        self.assertLess(logits.grad[1].item(),0)
        self.assertGreater(logits.grad[0].item(),0)
        self.assertIsNone(reference.grad)

    def test_proper_scoring_losses(self):
        logits, labels = torch.zeros(2), torch.tensor([0.,1.])
        self.assertAlmostEqual(proper_loss(logits,labels,"brier").item(),0.25)
        self.assertAlmostEqual(proper_loss(logits,labels,"log").item(),math.log(2),places=6)

    def test_exact_cpu_checkpoint_resume(self):
        full = train(self.config(4),self.directory)
        half = train(self.config(2),self.directory)
        resumed = train(self.config(4),self.directory,resume=half/"checkpoint.pt")
        a,b = read_checkpoint(full/"checkpoint.pt"),read_checkpoint(resumed/"checkpoint.pt")
        self.assertTrue(all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"]))
        self.assertTrue(torch.equal(a["torch_rng"],b["torch_rng"]))

    def test_posttraining_stages_are_separate_and_update(self):
        sft = train(self.config(),self.directory)
        pref = train(self.config(stage="preference"),self.directory,initialize=sft/"checkpoint.pt")
        rlvr = train(self.config(stage="rlvr"),self.directory,initialize=pref/"checkpoint.pt")
        records = [read_checkpoint(p/"checkpoint.pt") for p in (sft,pref,rlvr)]
        self.assertEqual([r["stage"] for r in records],["sft","preference","rlvr"])
        for a,b in zip(records,records[1:]):
            self.assertTrue(all(torch.equal(a["model"][k],b["reference"][k]) for k in a["model"]))
            self.assertTrue(any(not torch.equal(a["model"][k],b["model"][k]) for k in a["model"]))

    def test_stage_mismatch_cannot_resume(self):
        initial = train(self.config(),self.directory)
        with self.assertRaises(ContractError):
            train(self.config(4,"preference"),self.directory,resume=initial/"checkpoint.pt")

    def test_checkpoint_tampering_rejected(self):
        initial = train(self.config(1),self.directory)
        path = initial/"checkpoint.pt"
        with path.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaises(ContractError):
            read_checkpoint(path)

    def test_large_allocation_blocked_before_model_creation(self):
        cfg = self.config()
        cfg["model"]["layers"] = 10000
        with self.assertRaises(ContractError):
            train(cfg,self.directory)

    def test_preference_without_initial_checkpoint_fails(self):
        with self.assertRaises(ContractError):
            train(self.config(stage="preference"),self.directory)


if __name__ == "__main__":
    unittest.main()

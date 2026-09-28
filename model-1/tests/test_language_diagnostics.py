"""Fitting probe boundaries, scoring counterexamples and a real miniature study."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(HAS_TORCH, "torch/numpy required")
class LanguageDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        from aim.language_data import prepare
        from aim.language_diagnostics import freeze_probes
        from aim.tracking import ROOT
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = prepare("symbolic", "sft")
        self.probes = freeze_probes(self.data, 1)
        self.config = json.loads((ROOT/"configs/language-fitting.json").read_text())
        self.config.update(seeds=[17], tokenizers=["byte"], steps=[1, 2], probe_count=1, max_new_tokens=4)
        self.config["training"]["model"] = {"width": 16, "layers": 1, "heads": 2, "kv_heads": 1, "ffn_width": 32, "context": 256}
        self.config["training"]["checkpoint_every"] = 1

    def test_probe_integrity_and_holdout_rejection(self):
        from aim.contracts import ContractError
        from aim.language_diagnostics import freeze_probes, validate_probes
        from aim.tracking import digest
        bad = copy.deepcopy(self.probes)
        bad["rows"][0]["response"] = "altered"
        with self.assertRaises(ContractError): validate_probes(bad)
        bad = copy.deepcopy(self.probes)
        bad["rows"][0]["split"] = "test"
        bad["probe_hash"] = digest({k: v for k, v in bad.items() if k != "probe_hash"})
        with self.assertRaises(ContractError): validate_probes(bad)
        with self.assertRaises(ContractError): freeze_probes({**self.data, "holdout": []}, 1)
        with self.assertRaises(ContractError): freeze_probes(self.data, True)

    def test_symbolic_alternative_and_wrong_answer_are_distinct(self):
        from aim.language_diagnostics import score_text, divergence
        from aim.symbolic_data import response
        row = self.probes["rows"][0]
        alternative = response(row["lhs"])
        result = score_text("symbolic", row, alternative)
        self.assertTrue(result["syntax_valid"] and result["contract_valid"])
        self.assertEqual(result["check"]["outcome"], "PASS")
        self.assertNotEqual(alternative, row["response"])
        self.assertIsNotNone(divergence(list(row["response"].encode()), list(alternative.encode())))
        wrong = score_text("symbolic", row, response("0"))
        self.assertTrue(wrong["contract_valid"])
        self.assertEqual(wrong["check"]["outcome"], "FAIL")
        unknown = score_text("symbolic", row, response("x/0"))
        self.assertNotEqual(unknown["check"]["outcome"], "PASS")

    def test_json_offsets_duplicates_and_nonfinite_values(self):
        from aim.language_diagnostics import score_text
        row = self.probes["rows"][0]
        text = '{"α":1,}'
        result = score_text("symbolic", row, text)
        self.assertFalse(result["syntax_valid"])
        self.assertEqual(result["syntax_error_byte_offset"], result["syntax_error_character_offset"]+1)
        for text in ('{"rhs":"x","rhs":"y"}', '{"rhs":NaN}', '[]'):
            result = score_text("symbolic", row, text)
            self.assertFalse(result["contract_valid"])
            self.assertEqual(result["check"]["outcome"], "NOT_RUN")

    def test_numerical_checker_does_not_use_response_label(self):
        from aim.language_fit import structured_fixture
        from aim.language_data import prepare
        from aim.language_diagnostics import score_text
        from aim.tracking import Run
        with Run(self.root, "fixture", {}) as run:
            path = structured_fixture(run)
        row = prepare("structured", "sft", path)["train"][0]
        original = row["response"]
        row["response"] = "tampered reference label"
        correct = score_text("structured", row, original)
        self.assertEqual(correct["check"]["outcome"], "PASS")
        self.assertTrue(correct["check"]["target_agrees"])
        wrong = score_text("structured", row, '{"coefficients":[[9,9,9]],"evidence":["E0"]}')
        self.assertTrue(wrong["contract_valid"])
        self.assertEqual(wrong["check"]["outcome"], "FAIL")

    def test_divergence_counts_missing_extra_and_eos(self):
        from aim.language_diagnostics import divergence
        self.assertIsNone(divergence([65, 257], [65, 257]))
        self.assertEqual(divergence([65, 257], [65]), {"response_token_index": 1, "expected_id": 257, "observed_id": None})
        self.assertEqual(divergence([65], [65, 66])["expected_id"], None)
        self.assertEqual(divergence([65, 257], [65, 66])["response_token_index"], 1)

    def model(self):
        from aim.neural import CausalLM
        from aim.language_contract import model_config
        from aim.tokenization import ByteTokenizer
        from aim.training import seed_all
        seed_all(17)
        return CausalLM(model_config(self.config["training"]["model"], ByteTokenizer()))

    def test_measurement_restores_mode_rng_and_replays(self):
        import torch
        from aim.language_diagnostics import measure, audit_measurement
        from aim.tokenization import ByteTokenizer
        model = self.model()
        rng = torch.get_rng_state().clone()
        report = measure(model, ByteTokenizer(), self.probes, 4)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(report, measure(model, ByteTokenizer(), self.probes, 4))
        self.assertTrue(audit_measurement(report, self.probes, ByteTokenizer(), 256, 4))
        self.assertFalse(report["rows"][0]["coverage"]["reference_fits_budget"])

    def test_audit_rejects_changed_text_counts_and_summary(self):
        from aim.contracts import ContractError
        from aim.language_diagnostics import measure, audit_measurement
        from aim.tokenization import ByteTokenizer
        report = measure(self.model(), ByteTokenizer(), self.probes, 4)
        for change in (lambda r: r["rows"][0]["generation"].update(text="forged"),
                       lambda r: r["rows"][0]["teacher"].update(correct_tokens=999),
                       lambda r: r["summary"]["train"].update(contract_valid=999)):
            bad = copy.deepcopy(report)
            change(bad)
            with self.assertRaises(ContractError): audit_measurement(bad, self.probes, ByteTokenizer(), 256, 4)

    def test_context_rejection_preserves_mode(self):
        from aim.contracts import ContractError
        from aim.language_diagnostics import measure
        from aim.tokenization import ByteTokenizer
        model = self.model()
        with self.assertRaises(ContractError): measure(model, ByteTokenizer(), self.probes, 256)
        self.assertTrue(model.training)

    def test_eos_and_invalid_byte_diagnostics(self):
        import torch
        from types import SimpleNamespace
        from aim.language_diagnostics import measure, audit_measurement
        from aim.tokenization import ByteTokenizer
        class Constant(torch.nn.Module):
            def __init__(self, token):
                super().__init__()
                self.token = token
                self.cfg = SimpleNamespace(context=256, vocab_size=259)
            def forward(self, x):
                logits = torch.full((x.shape[0], x.shape[1], 259), -10.)
                logits[:, :, self.token] = 10.
                return logits
        for token, stop in ((257, "eos"), (258, "invalid_special_token"), (195, "invalid_utf8")):
            report = measure(Constant(token), ByteTokenizer(), self.probes, 1)
            self.assertTrue(audit_measurement(report, self.probes, ByteTokenizer(), 256, 1))
            self.assertEqual(report["rows"][0]["generation"]["termination"], stop)
            self.assertEqual(report["summary"]["train"]["scoped_check_pass"], 0)
            if token == 257:
                self.assertEqual(report["rows"][0]["teacher"]["correct_tokens"], 1)
                self.assertGreater(report["rows"][0]["teacher"]["gold_eos_probability"], .99)

    def test_nonfinite_model_failure_restores_mode(self):
        import torch
        from unittest.mock import patch
        from aim.contracts import ContractError
        from aim.language_diagnostics import measure
        from aim.tokenization import ByteTokenizer
        model = self.model()
        with patch.object(model, "forward", return_value=torch.full((1, 256, 259), float("nan"))):
            with self.assertRaises(ContractError): measure(model, ByteTokenizer(), self.probes, 4)
        self.assertTrue(model.training)

    def test_checkpoint_rejects_different_probe_dataset(self):
        from aim.contracts import ContractError
        from aim.language_training import train
        from aim.language_diagnostics import measure_checkpoint
        from aim.tracking import digest
        config = {**self.config["training"], "seed": 17, "steps": 1, "stage": "sft", "task": "symbolic"}
        run = train(config, self.root)
        bad = copy.deepcopy(self.probes)
        bad["dataset_hash"] = "0"*64
        bad["probe_hash"] = digest({k: v for k, v in bad.items() if k != "probe_hash"})
        with self.assertRaises(ContractError): measure_checkpoint(run/"checkpoint.pt", bad, 4)
        bad = copy.deepcopy(self.probes)
        bad["rows"][0]["response"] = "forged probe response"
        bad["probe_hash"] = digest({k: v for k, v in bad.items() if k != "probe_hash"})
        with self.assertRaises(ContractError): measure_checkpoint(run/"checkpoint.pt", bad, 4)

    def test_config_budgets_and_duplicates_reject(self):
        from aim.contracts import ContractError
        from aim.language_fit import validate_config
        validate_config(self.config)
        for change in ({"steps": [2, 1]}, {"steps": [1, 1]}, {"seeds": [True]}, {"tokenizers": ["byte", "byte"]}, {"max_new_tokens": 129}):
            with self.assertRaises(ContractError): validate_config({**self.config, **change})

    def test_real_study_export_and_tamper_detection(self):
        from aim.contracts import ContractError
        from aim.language_fit import run_study, audit_study, export
        from aim.corpus import read_json
        run = run_study(self.config, self.root/"runs")
        audited = audit_study(run)
        self.assertEqual(audited["arms"], 2)
        self.assertEqual(audited["measurements"], 6)
        self.assertEqual(audited["controller_replays"], 8)
        self.assertFalse(list(run.rglob("holdout.json")))
        export(run, self.root/"export")
        results = read_json(run/"results.json")
        bad = copy.deepcopy(results)
        bad["controls"].pop()
        (run/"results.json").write_text(json.dumps(bad))
        with self.assertRaises(ContractError): audit_study(run)
        (run/"results.json").write_text(json.dumps(results))
        path = run/results["arms"][0]["measurements"][0]["path"]
        path.write_text('{}')
        with self.assertRaises(ContractError): audit_study(run)


if __name__ == "__main__":
    unittest.main()

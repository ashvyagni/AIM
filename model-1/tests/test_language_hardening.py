"""Adversarial checks added during the Phase 3D audit."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from aim.contracts import ContractError
from aim.language_data import prepare
from aim.tracking import ROOT

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None


class LanguageBoundaryTests(unittest.TestCase):
    def test_external_rows_reject_undeclared_holdout_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            row = {"id": "a", "group": "a", "prompt": "Q?", "response": "A", "rights": "fixture",
                   "label_origin": "synthetic", "holdout_answers": ["not training data"]}
            record = {"schema": "aim-language-data-v1", "version": "fixture", "train": [row],
                      "validation": [{**row, "id": "b", "group": "b", "prompt": "R?"}]}
            path = Path(directory)/"data.json"; path.write_text(json.dumps(record))
            with self.assertRaises(ContractError): prepare("external", "sft", path)

    def test_symbolic_decode_failure_replaces_stale_trace(self):
        from aim.symbolic_loop import SymbolicTransformerResearcher
        researcher = SymbolicTransformerResearcher.__new__(SymbolicTransformerResearcher)
        model = SimpleNamespace(last_generation=None)
        def fail(*args):
            model.last_generation = {"termination": "invalid_utf8", "generated_ids": [195]}
            raise ContractError("invalid UTF-8")
        model.generate_text = fail
        researcher.model, researcher.tokenizer = model, None
        researcher.model_id, researcher.max_new_tokens = "fixture", 4
        researcher.last_trace = {"prompt": "stale earlier prompt", "raw": "earlier output"}
        with self.assertRaises(ContractError): researcher.hypothesize(SimpleNamespace(lhs="(x+9)**2"))
        self.assertIn("(x+9)**2", researcher.last_trace["prompt"])
        self.assertEqual(researcher.last_trace["generation"], model.last_generation)
        self.assertIsNone(researcher.last_trace["raw"])


@unittest.skipUnless(HAS_TORCH, "torch/numpy required")
class LanguageCheckpointTests(unittest.TestCase):
    def setUp(self):
        from aim.language_training import train
        from aim.language_contract import read_source
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = json.loads((ROOT/"configs/language-sft.json").read_text())
        self.config.update(steps=1)
        self.config["model"] = {"width":16,"layers":1,"heads":2,"kv_heads":1,"ffn_width":32,"context":128}
        self.run = train(self.config, self.root/"runs")
        self.record = read_source(self.run/"checkpoint.pt")

    def test_checkpoint_stable_architecture_must_match_saved_model(self):
        from aim.language_contract import validate_record
        record = copy.deepcopy(self.record)
        record["stable_config"]["model"]["width"] = 32
        with self.assertRaises(ContractError): validate_record(record)

    def test_nonfinite_frozen_reference_is_rejected(self):
        from aim.language_contract import validate_record
        record = copy.deepcopy(self.record)
        next(iter(record["reference"].values())).fill_(float("nan"))
        with self.assertRaises(ContractError): validate_record(record)


if __name__ == "__main__": unittest.main()

import copy
import json
import sqlite3
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

from aim.contracts import Action, ContractError, Decision, Outcome, Status, ToolResult, finite_number
from aim.controller import Controller
from aim.datasets import polynomial_case
from aim.evaluation import evaluate, load_suite
from aim.judge import TARGET, VerificationFirstJudge
from aim.memory import Memory
from aim.metrics import calibration_metrics
from aim.researcher import PolynomialResearcher
from aim.tool_worker import execute
from aim.tools import ToolRunner
from aim.tracking import ROOT, Run, canonical


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def loop(self, case=None, **options):
        with Run(self.directory, "test-loop", {}) as run:
            state = Controller(**options).run(case or polynomial_case("test", [1,2,1]), run)
        return state, run.path

    def memory(self):
        memory = Memory(self.directory / "memory")
        self.addCleanup(memory.close)
        return memory

    def source(self, memory, text="alpha observation", version="1"):
        return memory.ingest(text, uri="aim://test", title="alpha", version=version, rights="test-fixture")

    def test_full_loop_and_replay(self):
        state, path = self.loop()
        self.assertEqual([c.status for c in state.claims], [Status.CONTRADICTED, Status.VERIFIED])
        self.assertEqual(state.claims[1].predicted, 25)
        m = Memory(path / "memory")
        self.addCleanup(m.close)
        self.assertEqual(canonical(m.replay()), canonical(state.to_dict()))
        events = [json.loads(row[0])["kind"] for row in m.db.execute("SELECT record FROM events ORDER BY seq")]
        self.assertLess(events.index("FORECAST_BEFORE_MEASUREMENT"), events.index("VERIFICATION"))
        self.assertTrue(all(m.validate(ev) for ev in state.evidence))
        self.assertIn("not a proof of the global law", state.final_response)

    def test_unavailable_never_verified(self):
        state, _ = self.loop(polynomial_case("missing", [1,2,1], available=False))
        self.assertTrue(all(c.status == Status.UNKNOWN for c in state.claims))
        self.assertTrue(any(v.outcome == Outcome.UNKNOWN for v in state.verifications))

    def test_conflicting_evidence(self):
        state, _ = self.loop(polynomial_case("conflict", [1,2,1], conflicting=True))
        self.assertEqual(state.claims, [])
        self.assertTrue(state.contradictions)

    def test_budget_exhaustion(self):
        state, path = self.loop(max_actions=1)
        self.assertEqual(len(state.results), 1)
        self.assertEqual(state.claims[0].status, Status.UNKNOWN)
        self.assertIn("Action budget exhausted", state.unknowns)
        self.assertEqual(json.loads((path / "status.json").read_text())["status"], "COMPLETED")

    def test_forecast_cannot_verify(self):
        class Confident(VerificationFirstJudge):
            def decide(self, state, hypothesis, claim):
                return Decision(claim.id, 1.0, TARGET, "VERIFY", "overconfident-test", "uncalibrated")
        state, _ = self.loop(judge=Confident())
        self.assertEqual(state.claims[0].status, Status.CONTRADICTED)

    def test_abstention_is_unknown(self):
        class Abstain(VerificationFirstJudge):
            def decide(self, state, hypothesis, claim):
                return Decision(claim.id, 0.1, TARGET, "ABSTAIN", "test", "uncalibrated")
        state, _ = self.loop(judge=Abstain())
        self.assertTrue(all(c.status == Status.UNKNOWN for c in state.claims))
        self.assertFalse(state.verifications)

    def test_wrong_judge_claim_rejected(self):
        class Wrong(VerificationFirstJudge):
            def decide(self, state, hypothesis, claim):
                return Decision("other-claim", 1.0, TARGET, "VERIFY", "test", "uncalibrated")
        state, _ = self.loop(judge=Wrong())
        self.assertFalse(state.claims)
        self.assertIn("Judge forecast is bound", state.unknowns[0])

    def test_fabricated_citation_rejected(self):
        class Fabricator(PolynomialResearcher):
            def hypothesize(self, state):
                return [replace(super().hypothesize(state)[0], evidence_ids=("made-up",))]
        state, _ = self.loop(researcher=Fabricator())
        self.assertFalse(state.claims)
        self.assertIn("Researcher fabricated", state.unknowns[0])

    def test_model_cannot_mutate_owned_state(self):
        class Mutator(PolynomialResearcher):
            def plan(self, state):
                state.target_x = 999
                return super().plan(state)
        state, _ = self.loop(researcher=Mutator())
        self.assertEqual(state.target_x, 4)

    def test_verified_claim_mutation_detected(self):
        state, path = self.loop()
        state.claims[1].predicted += 1
        m = Memory(path / "memory")
        self.addCleanup(m.close)
        with self.assertRaises(ContractError):
            Controller._validate_final(state, m)

    def test_nested_records_serialize(self):
        result = ToolResult("a", Outcome.PASS, 1, "test", 0.1, "1")
        self.assertEqual(json.loads(canonical([result]))[0]["outcome"], "PASS")

    def test_changed_source_detected(self):
        m = self.memory()
        ev = m.span(self.source(m), 0, 5)
        (m.objects / ev.source_hash).write_text("altered")
        with self.assertRaises(ContractError):
            m.validate(ev)

    def test_forged_quote_rejected(self):
        m = self.memory()
        ev = m.span(self.source(m), 0, 5)
        self.assertFalse(m.validate(replace(ev, quote="fake!")))

    def test_source_versions_are_distinct(self):
        m = self.memory()
        self.assertNotEqual(self.source(m, version="1"), self.source(m, version="2"))

    def test_ledger_append_only(self):
        m = self.memory()
        m.append("STATE", {"phase":"first"})
        with self.assertRaises(sqlite3.IntegrityError):
            m.db.execute("DELETE FROM events")

    def test_ledger_hash_detects_tampering(self):
        m = self.memory()
        m.append("STATE", {"phase":"first"})
        m.db.execute("DROP TRIGGER events_no_update")
        m.db.execute("UPDATE events SET record='{}'")
        with self.assertRaises(ContractError):
            m.replay()

    def test_calculation_at_zero(self):
        result = execute({"operation":"CALCULATE","arguments":{"coefficients":[2,3,4],"x":0}})
        self.assertEqual(result["value"], 2)

    def test_actual_worker_timeout(self):
        result = ToolRunner().execute(Action("tiny-deadline", "CALCULATE", {"coefficients":[1],"x":1}, 1e-9))
        self.assertEqual(result.outcome, Outcome.TIMEOUT)

    def test_tool_allowlist(self):
        with self.assertRaises(ContractError):
            ToolRunner().execute(Action("bad", "EXEC_SHELL", {}))

    def test_nonfinite_and_boolean_rejected(self):
        for value in (True, float("nan"), float("inf"), "1"):
            with self.subTest(value=value), self.assertRaises(ContractError):
                finite_number(value)

    def test_unknown_not_scored_false(self):
        metrics = calibration_metrics([0.9, 0.2], [1, None])
        self.assertEqual(metrics["resolved"], 1)
        self.assertAlmostEqual(metrics["brier"], 0.01)

    def test_canonical_suite(self):
        _, metrics = evaluate(ROOT / "eval/suite-v1.json", self.directory)
        self.assertEqual(metrics["passed"], 8)

    def test_eval_change_rejected(self):
        path = self.directory / "suite.json"
        path.write_text('{}')
        path.with_suffix(".sha256").write_text('0'*64)
        with self.assertRaises(ContractError):
            load_suite(path)

    def test_failure_is_retained_with_source_snapshot(self):
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with Run(self.directory, "failed-test", {}) as run:
                raise RuntimeError("intentional")
        self.assertEqual(json.loads((run.path / "status.json").read_text())["status"], "FAILED")
        self.assertIn("intentional", (run.path / "failure.txt").read_text())
        with zipfile.ZipFile(run.path / "source-snapshot.zip") as archive:
            self.assertIn("aim/controller.py", archive.namelist())


if __name__ == "__main__":
    unittest.main()

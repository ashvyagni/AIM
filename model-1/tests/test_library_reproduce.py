import copy
import tempfile
import unittest
from pathlib import Path

from aim.contracts import ContractError
from aim.library_reproduce import benchmark, crash_rollback, expected_failures, lifecycle, validate_config
from aim.tracking import Run


class LibraryStudyTests(unittest.TestCase):
    def config(self):
        return {"schema": "aim-library-study-v1", "sizes": [16], "queries": 1, "repeats": 1,
                "chunk_size": 512, "overlap": 64, "limit": 5, "per_source": 1}

    def test_registered_config_bounds(self):
        config = self.config()
        validate_config(config)
        for field, value in (("sizes", [1025]), ("repeats", True), ("limit", True), ("queries", 17), ("overlap", 300)):
            changed = copy.deepcopy(config)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ContractError):
                validate_config(changed)

    def test_small_benchmark_reports_actual_query_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = benchmark(Path(tmp), self.config())
            self.assertEqual(len(result["queries"]), 1)
            self.assertEqual(result["queries"][0]["rank"], 1)
            self.assertEqual(result["results"][0]["recall_at_5"], 1)
            self.assertGreater(result["results"][0]["database_bytes"], 0)
            self.assertEqual(result["results"][0]["negative_control_hits"], 0)

    def test_lifecycle_reproduction(self):
        with tempfile.TemporaryDirectory() as tmp, Run(Path(tmp), "test-library-lifecycle", {}) as run:
            result = lifecycle(run)
            self.assertTrue(result["cross_run_state_equal"])
            self.assertEqual(len(result["reviews"]), 3)
            self.assertEqual(result["final"]["statuses"]["RETRACTED"], 1)

    def test_crash_and_failure_retention_reproduction(self):
        with tempfile.TemporaryDirectory() as tmp, Run(Path(tmp), "test-library-failures", {}) as run:
            result = crash_rollback(run)
            self.assertEqual(result["exit_code"], 23)
            self.assertTrue(result["unchanged"])
            failed = expected_failures(run)
            self.assertEqual(len(failed), 2)


if __name__ == "__main__":
    unittest.main()

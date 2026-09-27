import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from aim.audit_contracts import seal, verify
from aim.contracts import ContractError
from aim.corpus import Corpus, intake
from aim.corpus_fixture import create_fixture
from aim.corpus_release import audit_release, export_subset, gate, review_template
from aim.corpus_review import DEFAULT, audit, audit_run, shingles, validate_audit
from aim.fleet_plan import build_plan, policy_template
from aim.hardware_audit import SCHEMA, collect, collect_run, observed, proc_cpu, proc_memory, storage_probe, validate_node
from aim.tracking import ROOT, file_hash, write_json

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None


def reseal(value):
    return seal({k: v for k, v in value.items() if k != "record_hash"})


def simulated_node(node_id="simulated-1"):
    values = {"logical_cpus": 8, "physical_cores": 4, "ram_bytes": 32*1024**3,
              "available_ram_bytes": 16*1024**3, "free_disk_bytes": 100*1024**3,
              "cpu_model": "SIMULATED CPU", "os": "SIMULATED OS", "architecture": "SIMULATED",
              "python": "3.12.14", "torch": "2.8.0"}
    return seal({"schema": SCHEMA, "node_id": node_id, "observed_at_unix": 1000,
                 "measurements": {k: observed(v, "synthetic unit-test fixture") for k, v in values.items()},
                 "scope": "SIMULATED test record; not a measured host"})


def simulated_policy(nodes):
    value = policy_template(nodes)
    value.update(reviewer="synthetic test", basis="unit-test declarations only", valid_until_unix=2000,
                 memory_budget_bytes_per_node=1024**3, disk_budget_bytes_per_node=1024**3)
    value["permissions"] = {k: True for k in value["permissions"]}
    return seal(value)


def fixture_review(report, accepted=None):
    value = review_template(report)
    value.update(purpose="engineering_fixture", reviewer="fixture-construction-v1", basis="Project-generated text; demonstrates review plumbing only")
    for row in value["documents"]:
        row.update(decision="accept" if accepted is None or row["id"] in accepted else "quarantine",
                   reason="Explicit fixture selection for systems validation", rights_evidence="Project fixture generator; no external passages",
                   privacy_evidence="Project fixture construction; no person records")
    return seal(value)


class HardwareAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = json.loads((ROOT/"configs/distributed-pretrain-smoke.json").read_text())

    def tearDown(self):
        self.tmp.cleanup()

    def test_linux_memory_units_and_bad_format(self):
        self.assertEqual(proc_memory("MemTotal: 123 kB\nMemAvailable: 45 kB\n"), {"MemTotal": 125952, "MemAvailable": 46080})
        for value in ("MemTotal: -1 kB", "MemTotal: 4 MB", "MemTotal: 3"):
            with self.assertRaises(ContractError): proc_memory(value)

    def test_linux_physical_pairs_and_missing_cpu_ids(self):
        raw = "model name : Test\nphysical id : 0\ncore id : 1\n\nmodel name : Test\nphysical id : 0\ncore id : 1\n\nmodel name : Test\nphysical id : 1\ncore id : 1"
        self.assertEqual(proc_cpu(raw), ("Test", 2))
        self.assertEqual(proc_cpu("processor : 0"), (None, None))

    def test_real_local_observation_portable_and_hash_bound(self):
        value = collect("local-test", self.root)
        self.assertEqual(validate_node(value), value)
        self.assertNotIn(str(Path.home()), json.dumps(value))
        self.assertNotIn("hostname", value)
        value["node_id"] = "modified"
        with self.assertRaises(ContractError): validate_node(value)

    def test_invalid_node_values_and_extra_fields(self):
        for key, value in (("logical_cpus", True), ("ram_bytes", -1), ("available_ram_bytes", 33*1024**3), ("physical_cores", 100)):
            node = simulated_node()
            node["measurements"][key]["value"] = value
            with self.assertRaises(ContractError): validate_node(reseal(node))
        node = simulated_node(); node["secret"] = "not allowed"
        with self.assertRaises(ContractError): validate_node(reseal(node))
        with self.assertRaises(ContractError): collect("../path", self.root)

    @unittest.skipUnless(HAS_TORCH,"torch/numpy required for model trial planning")
    def test_missing_measurement_stays_unknown(self):
        node = simulated_node()
        node["measurements"]["available_ram_bytes"] = observed(None, "not provided")
        node = reseal(node)
        result = build_plan([node], self.config, simulated_policy([node]), now=1001)
        self.assertIn("available_memory_unknown:simulated-1", result["blockers"])
        self.assertEqual(result["status"], "BLOCKED")

    @unittest.skipUnless(HAS_TORCH,"torch/numpy required for model trial planning")
    def test_zero_available_is_observation_not_unknown(self):
        node = simulated_node(); node["measurements"]["available_ram_bytes"]["value"] = 0
        node = reseal(node)
        result = build_plan([node], self.config, simulated_policy([node]), now=1001)
        self.assertIn("memory_budget_exceeds_observed_availability:simulated-1", result["blockers"])

    @unittest.skipUnless(HAS_TORCH,"torch/numpy required for model trial planning")
    def test_duplicate_stale_and_future_nodes(self):
        node = simulated_node()
        with self.assertRaises(ContractError): build_plan([node,node], self.config, now=1001)
        result = build_plan([node], self.config, now=90000)
        self.assertIn("stale_or_future_observation:simulated-1", result["blockers"])
        result = build_plan([node], self.config, now=100)
        self.assertIn("stale_or_future_observation:simulated-1", result["blockers"])

    @unittest.skipUnless(HAS_TORCH,"torch/numpy required for model trial planning")
    def test_policy_binding_expiration_permissions_and_capacity(self):
        nodes = [simulated_node()]
        policy = simulated_policy(nodes)
        policy["node_hashes"] = []
        with self.assertRaises(ContractError): build_plan(nodes, self.config, reseal(policy), now=1001)
        policy = simulated_policy(nodes); policy["valid_until_unix"] = 1000
        policy["permissions"]["network_ports"] = False
        policy["memory_budget_bytes_per_node"] = 1
        policy["disk_budget_bytes_per_node"] = 1
        result = build_plan(nodes, self.config, reseal(policy), now=1001)
        for blocker in ("operator_policy_expired", "permission_missing:network_ports", "operator_memory_budget_below_parameter_state_floor", "operator_disk_budget_below_checkpoint_planning_floor"):
            self.assertIn(blocker, result["blockers"])

    @unittest.skipUnless(HAS_TORCH,"torch/numpy required for model trial planning")
    def test_simulated_four_host_plan_does_not_claim_deployment(self):
        nodes = [simulated_node("simulated-"+str(i)) for i in range(4)]
        result = build_plan(nodes, self.config, simulated_policy(nodes), now=1001)
        self.assertEqual(result["status"], "READY_FOR_OPERATOR_TRIAL")
        self.assertEqual([r["physical_nodes"] for r in result["trials"]], [1,2,4])
        self.assertEqual([r["global_sequences_per_update"] for r in result["trials"]], [4,8,16])
        self.assertIn("physical_host_distinctness", result["unmeasured"])
        self.assertEqual(result["fp32_state_floor_bytes_per_rank"],16*6496)

    @unittest.skipUnless(HAS_TORCH,"torch/numpy required for model trial planning")
    def test_heterogeneous_or_missing_runtime_blocks(self):
        one, two = simulated_node(), simulated_node("simulated-2")
        two["measurements"]["torch"] = observed(None,"not installed")
        two = reseal(two)
        result = build_plan([one,two],self.config,now=1001)
        self.assertIn("missing_runtime_identity",result["blockers"])
        self.assertIn("heterogeneous_runtime_requires_separate_trial",result["blockers"])

    def test_bounded_real_storage_probe_and_source_observation(self):
        path = collect_run("local-test",self.root,self.root/"runs")
        probe = storage_probe(path,self.root/"runs",1)
        value = json.loads(probe.read_text())
        self.assertTrue(value["round_trip_equal"])
        self.assertEqual(value["bytes"],1024**2)
        self.assertEqual(file_hash(probe.parent/"probe.bin"),value["sha256"])
        self.assertEqual(value["node_hash"],json.loads(path.read_text())["record_hash"])
        with self.assertRaises(ContractError): storage_probe(path,self.root/"runs",33)

    def test_storage_probe_rejects_foreign_host(self):
        path=self.root/"node.json";write_json(path,simulated_node())
        with self.assertRaisesRegex(ContractError,"probe host differs"):
            storage_probe(path,self.root/"runs",1)
        self.assertEqual(json.loads(next((self.root/"runs").glob("*/status.json")).read_text())["status"],"FAILED")

    def test_unsupported_platform_does_not_guess_ram_or_cpu(self):
        with patch("aim.hardware_audit.platform.system",return_value="Unsupported"):
            node=collect("unknown-platform",self.root)
        self.assertIsNone(node["measurements"]["ram_bytes"]["value"])
        self.assertIsNone(node["measurements"]["cpu_model"]["value"])


class CorpusReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.path=intake(create_fixture(self.root/"fixture"),self.root/"intake")
        self.corpus=Corpus(self.path);self.report=audit(self.corpus)

    def tearDown(self): self.tmp.cleanup()

    def test_overlap_detects_actual_template_family_leakage(self):
        self.assertGreater(self.report["cross_split_pairs"],0)
        self.assertEqual(validate_audit(self.report,self.corpus),self.report)
        self.assertEqual(self.report,audit(self.corpus))
        self.assertEqual(self.report["counts_by_split_and_type"]["train"],dict(code=3,math=3,prose=3,unicode=3))

    def test_unicode_normalization_and_short_text(self):
        self.assertEqual(shingles("CAFÉ α",2),shingles("cafe\u0301 Α",2))
        self.assertEqual(shingles("x",5),{("x",)})
        self.assertEqual(shingles("  ",5),set())

    def test_all_audit_budgets_fail_without_partial_clean_report(self):
        for key in ("max_documents","max_total_shingles","max_candidate_pairs","max_pair_updates"):
            with self.subTest(key=key),self.assertRaises(ContractError): audit(self.corpus,{**DEFAULT,key:1})

    def test_invalid_thresholds_and_config_fields(self):
        for value in (True,0,1.1,float("nan")):
            with self.assertRaises(ContractError): audit(self.corpus,{**DEFAULT,"jaccard_threshold":value})
        with self.assertRaises(ContractError): audit(self.corpus,{**DEFAULT,"extra":1})

    def test_rehashed_false_report_does_not_pass_replay(self):
        value=copy.deepcopy(self.report);value["pairs"]=[];value["cross_split_pairs"]=0
        with self.assertRaises(ContractError): validate_audit(reseal(value),self.corpus)

    def test_inverted_index_matches_brute_force_pair_scores(self):
        rows=sorted(self.corpus.index["documents"],key=lambda r:r["id"])
        features=[shingles(self.corpus.text(r),DEFAULT["shingle_width"]) for r in rows]
        expected=[]
        for i in range(len(rows)):
            for j in range(i+1,len(rows)):
                shared=len(features[i]&features[j])
                jac=shared/len(features[i]|features[j])
                containment=shared/min(len(features[i]),len(features[j]))
                if shared>=DEFAULT["minimum_shared_shingles"] and (jac>=DEFAULT["jaccard_threshold"] or containment>=DEFAULT["containment_threshold"]):
                    expected.append((rows[i]["id"],rows[j]["id"],shared,jac,containment))
        self.assertEqual(expected,[(p["left"],p["right"],p["shared_shingles"],p["jaccard"],p["containment"]) for p in self.report["pairs"]])

    def test_changed_source_object_is_not_a_clean_audit(self):
        row=self.corpus.index["documents"][0]
        (self.path.parent/"objects"/row["sha256"]).write_text("modified")
        with self.assertRaises(ContractError):audit(self.corpus)

    def test_review_is_hash_bound_complete_and_evidence_required(self):
        for change in ("hash","membership","evidence"):
            review=fixture_review(self.report)
            if change=="hash":review["audit_hash"]="0"*64
            if change=="membership":review["documents"][0]=review["documents"][1]
            if change=="evidence":review["documents"][0]["rights_evidence"]=None
            with self.assertRaises(ContractError):gate(self.corpus,self.report,reseal(review))

    def test_accept_all_is_blocked_by_cross_split_pairs(self):
        result=gate(self.corpus,self.report,fixture_review(self.report))
        self.assertEqual(result["status"],"BLOCKED")
        self.assertTrue(all(b.startswith("accepted_cross_split_overlap:") for b in result["blockers"]))

    def test_unreviewed_and_missing_splits_block(self):
        review=fixture_review(self.report,set())
        review["documents"][0]["decision"]="unreviewed"
        result=gate(self.corpus,self.report,reseal(review))
        self.assertIn("missing_accepted_split:train",result["blockers"])
        self.assertIn("missing_accepted_split:validation",result["blockers"])
        self.assertTrue(any(b.startswith("unreviewed:") for b in result["blockers"]))

    def release(self):
        accepted={"document-000","document-004","document-008","document-014","document-019"}
        a=self.root/"audit.json";r=self.root/"review.json"
        write_json(a,self.report);write_json(r,fixture_review(self.report,accepted))
        return export_subset(self.path,a,r,self.root/"exports"),a

    def test_release_preserves_parent_and_replays_decisions(self):
        before={p.name:file_hash(p) for p in (self.path.parent/"objects").iterdir()}
        release,a=self.release()
        result=audit_release(release,self.path,a)
        self.assertEqual((result["accepted"],result["quarantined"]),(5,15))
        self.assertTrue(result["original_bytes_and_metadata_preserved"])
        self.assertEqual(before,{p.name:file_hash(p) for p in (self.path.parent/"objects").iterdir()})

    def test_blocked_export_retains_failed_gate(self):
        a=self.root/"audit.json";r=self.root/"review.json"
        write_json(a,self.report);write_json(r,fixture_review(self.report))
        with self.assertRaisesRegex(ContractError,"export blocked"):
            export_subset(self.path,a,r,self.root/"exports")
        run=next((self.root/"exports").iterdir())
        self.assertEqual(json.loads((run/"status.json").read_text())["status"],"FAILED")
        self.assertEqual(json.loads((run/"gate.json").read_text())["status"],"BLOCKED")
        self.assertFalse((run/"release.json").exists())

    def test_release_tamper_and_traversal_fail(self):
        release,a=self.release();original=json.loads(release.read_text())
        for change in ("purpose","path","metadata"):
            value=copy.deepcopy(original)
            if change=="purpose":value["purpose"]="production_candidate"
            if change=="path":value["corpus_path"]="../other.json"
            if change=="metadata":value["accepted"]=[]
            release.write_text(json.dumps(reseal(value)))
            with self.assertRaises(ContractError):audit_release(release,self.path,a)

    def test_cli_audit_gate_and_record_seal(self):
        result=subprocess.run([sys.executable,"-m","aim","corpus-audit","--corpus",str(self.path),"--runs",str(self.root/"audits")],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        a=Path(result.stdout.strip())
        review=fixture_review(json.loads(a.read_text()))
        raw=self.root/"raw.json";sealed=self.root/"sealed.json"
        write_json(raw,{k:v for k,v in review.items() if k!="record_hash"})
        result=subprocess.run([sys.executable,"-m","aim","audit-seal","--input",str(raw),"--output",str(sealed)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        result=subprocess.run([sys.executable,"-m","aim","corpus-gate","--corpus",str(self.path),"--audit",str(a),"--review",str(sealed),"--runs",str(self.root/"gates")],capture_output=True,text=True)
        self.assertEqual(result.returncode,2,result.stderr)


if __name__=="__main__":unittest.main()

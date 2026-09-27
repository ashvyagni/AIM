"""Reproduce operator/data readiness without inventing lab measurements or approvals."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .audit_contracts import seal, verify
from .contracts import ContractError
from .corpus import Corpus, intake, read_json, require
from .corpus_fixture import create_fixture
from .corpus_release import audit_release, export_subset, review_template
from .corpus_review import audit, audit_run
from .distributed_jobs import audit_job, launch
from .fleet_plan import build_plan, plan_run
from .hardware_audit import collect_run, storage_probe, validate_node
from .tracking import ROOT, Run, file_hash, write_json


def construction_review(report, selected=None):
    """Explicit procedural review of our generated fixture, never external data."""
    result = review_template(report)
    result.update(purpose="engineering_fixture", reviewer="fixture-construction-v1",
                  basis="Authored integration fixture; selected subset demonstrates review/export, not a training mixture recommendation")
    for row in result["documents"]:
        row.update(decision="accept" if selected is None or row["id"] in selected else "quarantine",
                   reason="All-documents negative control" if selected is None else "Fixed train-prose/validation-math/test-Unicode demonstration selection",
                   rights_evidence="aim.corpus_fixture source; project-generated text without external passages",
                   privacy_evidence="Fixture construction; no real-person records")
    return seal(result)


def reproduce(runs):
    config_path = ROOT/"configs/distributed-pretrain-smoke.json"
    with Run(Path(runs), "phase-3c-build", {"scope": "local operator/corpus infrastructure; no lab deployment or external-data approval",
             "selection_rule": "train prose; validation math; test unicode", "storage_probe_mebibytes": 4}, [config_path]) as run:
        tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (run.path/"tests.log").write_text(tests.stdout)
        require(tests.returncode == 0 and "skipped" not in tests.stdout.lower(), "full regression failed/skipped; retained tests.log")
        print("Full regression passed with no skips", flush=True)
        node_path = collect_run("local-development", ROOT, run.path/"hardware")
        node = read_json(node_path)
        probe_path = storage_probe(node_path, run.path/"hardware", 4)
        plan_path = plan_run([node_path], config_path, run.path/"hardware")
        require(read_json(plan_path)["status"] == "BLOCKED", "no operator policy supplied; trial must remain blocked")
        # Retain an actual failed import, not a successful plan with a hidden repaired ID.
        bad_path = run.path/"tampered-node.json"
        write_json(bad_path, {**node, "node_id": "changed-without-rehash"})
        try:
            plan_run([bad_path], config_path, run.path/"invalid-node-import")
        except ContractError as exc:
            require("hash mismatch" in str(exc), "unexpected import rejection")
        else:
            raise RuntimeError("tampered observation imported")
        print("Local hardware/storage collected; missing policy and tampered record rejected", flush=True)

        corpus_path = intake(create_fixture(run.path/"fixture"), run.path/"intake")
        corpus = Corpus(corpus_path)
        before = {p.name: file_hash(p) for p in (corpus_path.parent/"objects").iterdir()}
        audit_path = audit_run(corpus_path, run.path/"audits")
        report = read_json(audit_path, 16*1024*1024)
        require(report["cross_split_pairs"] > 0, "fixture negative-control overlap not detected")
        all_review = run.path/"accept-all-review.json"
        write_json(all_review, construction_review(report))
        try:
            export_subset(corpus_path, audit_path, all_review, run.path/"blocked-export")
        except ContractError as exc:
            require("export blocked" in str(exc), "unexpected subset rejection")
        else:
            raise RuntimeError("overlapping cross-split export succeeded")
        selected = {r["id"] for r in corpus.index["documents"] if
                    (r["split"], r["content_type"]) in {("train", "prose"), ("validation", "math"), ("test", "unicode")}}
        review_path = run.path/"subset-review.json"
        write_json(review_path, construction_review(report, selected))
        release_path = export_subset(corpus_path, audit_path, review_path, run.path/"release")
        certificate = audit_release(release_path, corpus_path, audit_path)
        write_json(run.path/"release-audit.json", certificate)
        require(before == {p.name: file_hash(p) for p in (corpus_path.parent/"objects").iterdir()}, "original corpus changed")
        release = read_json(release_path)
        released_path = release_path.parent/release["corpus_path"]
        released_audit = audit(Corpus(released_path))
        write_json(run.path/"released-overlap-audit.json", released_audit)
        require(released_audit["cross_split_pairs"] == 0, "reviewed subset still crosses lexical gate")
        print("Known overlap blocked; reviewed five-document subset exported and independently replayed", flush=True)
        config = {**read_json(config_path), "steps": 2}
        trained = launch(config, released_path, run.path/"training")
        measured = audit_job(trained, released_path)
        print("Reviewed corpus completed actual two-worker training and data/checkpoint audit", flush=True)
        failed = []
        for group in ("invalid-node-import", "blocked-export"):
            child = next((run.path/group).iterdir())
            status = read_json(child/"status.json")
            require(status["status"] == "FAILED", "negative control failure not retained")
            failed.append({"path": str(child.relative_to(run.path)), "status": status,
                           "failure_sha256": file_hash(child/"failure.txt")})
        write_json(run.path/"results.json", {"node_path": str(node_path.relative_to(run.path)), "probe_path": str(probe_path.relative_to(run.path)),
                   "plan_path": str(plan_path.relative_to(run.path)), "parent_corpus": str(corpus_path.relative_to(run.path)),
                   "audit_path": str(audit_path.relative_to(run.path)), "review_path": str(review_path.relative_to(run.path)),
                   "release_path": str(release_path.relative_to(run.path)), "release_audit": certificate,
                   "parent_objects_unchanged": True, "failed_controls": failed,
                   "training": {"path": str(trained.relative_to(run.path)), "audit": measured,
                                "summary": read_json(trained/"job/summary.json"), "status": read_json(trained/"status.json")},
                   "scope": "engineering fixture and one local host; no corpus quality, physical scaling or model capability conclusion"})
    return run.path


def export(path, destination):
    path, destination = Path(path), Path(destination)
    require(read_json(path/"status.json")["status"] == "COMPLETED", "cannot export incomplete reproduction")
    result = read_json(path/"results.json")
    certificate = audit_release(path/result["release_path"], path/result["parent_corpus"], path/result["audit_path"])
    require(certificate == result["release_audit"], "release export audit changed")
    node = validate_node(read_json(path/result["node_path"]))
    plan = read_json(path/result["plan_path"])
    require(build_plan([node], plan["configuration"], now=plan["created_at_unix"]) == plan, "fleet preparation record changed")
    release_path = path/result["release_path"]
    release = read_json(release_path)
    corpus_path = release_path.parent/release["corpus_path"]
    require(audit(Corpus(corpus_path)) == read_json(path/"released-overlap-audit.json"), "released overlap evidence changed")
    require(audit_job(path/result["training"]["path"], corpus_path) == result["training"]["audit"], "training export audit changed")
    for failure in result["failed_controls"]:
        require(file_hash(path/failure["path"]/"failure.txt") == failure["failure_sha256"], "retained failure changed")
    probe = read_json(path/result["probe_path"])
    verify(probe, "aim-storage-probe-v1")
    live = validate_node(read_json((path/result["probe_path"]).parent/"live-node.json"))
    require(probe["live_node_hash"] == live["record_hash"], "probe live observation changed")
    require(probe["node_hash"] == node["record_hash"] and file_hash((path/result["probe_path"]).parent/"probe.bin") == probe["sha256"], "storage evidence changed")
    destination.mkdir(parents=True, exist_ok=False)
    files = {"results.json": path/"results.json", "tests.log": path/"tests.log", "node.json": path/result["node_path"],
             "storage-probe.json": path/result["probe_path"], "fleet-plan.json": path/result["plan_path"],
             "parent-overlap-audit.json": path/result["audit_path"], "review-decisions.json": path/result["review_path"],
             "release.json": release_path, "release-audit.json": path/"release-audit.json",
             "released-overlap-audit.json": path/"released-overlap-audit.json"}
    for name, source in files.items():
        shutil.copyfile(source, destination/name)
    manifest = read_json(path/"manifest.json")
    write_json(destination/"manifest.json", {"run_id": path.name, "git_commit": manifest["environment"]["git_commit"],
               "code_hash": manifest["code_hash"], "code_hashes": manifest["code_hashes"], "configuration": manifest["configuration"],
               "environment": {k: manifest["environment"][k] for k in ("python", "platform", "machine", "logical_cpus", "packages")},
               "status": read_json(path/"status.json"), "files": {name: file_hash(destination/name) for name in files},
               "scope": "portable records; original referenced paths are run-relative and resolved in retained local run, not this export"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    path = reproduce(args.runs)
    if args.export:
        export(path, args.export)
    print(path)


if __name__ == "__main__":
    main()

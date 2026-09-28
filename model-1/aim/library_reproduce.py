"""Registered local-library engineering study. No external papers or truth labels."""
import argparse
import json
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .corpus import read_json, require
from .evidence_review import EvidenceReviewController, ExcerptProposal, ExtractiveResearcher, audit_review
from .library import Library, identity
from .library_cli import ingest_manifest
from .tracking import ROOT, Run, digest, file_hash, write_json


def validate_config(c):
    require(isinstance(c, dict) and set(c) == {"schema", "sizes", "queries", "repeats", "chunk_size", "overlap", "limit", "per_source"}
            and c["schema"] == "aim-library-study-v1", "invalid library study configuration")
    require(isinstance(c["sizes"], list) and 1 <= len(c["sizes"]) <= 4
            and all(type(n) is int and 16 <= n <= 1024 for n in c["sizes"])
            and c["sizes"] == sorted(set(c["sizes"])), "invalid benchmark sizes")
    require(type(c["queries"]) is int and 1 <= c["queries"] <= 16 and type(c["repeats"]) is int
            and 1 <= c["repeats"] <= 10 and c["limit"] == 5 and type(c["limit"]) is int
            and c["per_source"] == 1 and type(c["per_source"]) is int, "invalid benchmark query budget")
    identity("fixture", uri="fixture:config", version="1", title="fixture", rights="AIM fixture",
             chunk_size=c["chunk_size"], overlap=c["overlap"])


def synthetic_documents(count, config):
    rows = []
    for i in range(count):
        term = f"specimen{i:05d}"
        text = "\n".join(f"Section {j}. The authored {term} note discusses instrument readings, calibration, uncertainty and recorded provenance. "
                         "It contains no measured scientific result and provides no evidence about model capabilities."
                         for j in range(8))
        rows.append({"text": text, "uri": "fixture:" + term, "version": "1", "title": term + " authored note",
                     "rights": "AIM authored engineering fixture; retrieval allowed", "chunk_size": config["chunk_size"], "overlap": config["overlap"]})
    return rows


def benchmark(path, config):
    results, query_records = [], []
    for size in config["sizes"]:
        database = path / f"benchmark-{size}.sqlite"
        docs = synthetic_documents(size, config)
        with Library(database, create=True) as library:
            started = time.perf_counter()
            ids = []
            for offset in range(0, size, 100):
                ids.extend(library.ingest_documents(docs[offset:offset+100]))
            ingest_seconds = time.perf_counter() - started
            started = time.perf_counter()
            audit = library.audit()
            audit_seconds = time.perf_counter() - started
            queries = [{"query": f"specimen{i:05d} calibration", "expected_source": ids[i]} for i in range(config["queries"])]
            for q in queries:
                library.search(q["query"], limit=config["limit"], per_source=config["per_source"])
            latencies, recalls, reciprocal_ranks = [], [], []
            for repeat in range(config["repeats"]):
                for q in queries:
                    started = time.perf_counter()
                    packet = library.search(q["query"], limit=config["limit"], per_source=config["per_source"])
                    elapsed = time.perf_counter() - started
                    library.verify_packet(packet)  # outside measured search latency
                    returned = [h["source_id"] for h in packet["hits"]]
                    rank = returned.index(q["expected_source"]) + 1 if q["expected_source"] in returned else None
                    latencies.append(elapsed)
                    recalls.append(int(rank is not None))
                    reciprocal_ranks.append(1 / rank if rank else 0)
                    query_records.append({"size": size, "repeat": repeat, **q, "returned_sources": returned,
                                          "rank": rank, "elapsed_seconds": elapsed, "packet_hash": packet["packet_hash"]})
            negative = library.search("absentvocabularytoken", limit=config["limit"], per_source=1)
            require(not negative["hits"], "negative retrieval control returned evidence")
            results.append({"versions": size, "fixture_hash": digest(docs), "snapshot": audit["snapshot"],
                            "original_bytes": audit["bytes"], "chunks": audit["chunks"], "database_bytes": database.stat().st_size,
                            "ingest_seconds": ingest_seconds, "audit_seconds": audit_seconds, "measured_queries": len(latencies),
                            "median_search_seconds": statistics.median(latencies),
                            "p95_search_seconds": sorted(latencies)[max(0, (95*len(latencies)+99)//100-1)],
                            "recall_at_5": statistics.mean(recalls), "mrr_at_5": statistics.mean(reciprocal_ranks),
                            "negative_control_hits": len(negative["hits"])})
    return {"schema": "aim-library-benchmark-v1", "sqlite_version": sqlite3.sqlite_version,
            "scope": "authored exact-keyword fixtures on one local host; not semantic retrieval quality or cluster performance",
            "timing": "one warmup per query; search includes ledger validation and returned-source checks; packet replay excluded",
            "results": results, "queries": query_records}


def lifecycle(parent):
    database = parent.path / "lifecycle.sqlite"
    with Library(database, create=True) as library:
        sid = library.ingest("An authored calibration note reports uncertain measurements. No scientific finding is claimed.",
                             uri="fixture:study", version="1", title="Calibration note", rights="AIM fixture")
        initial = library.audit()
    first = EvidenceReviewController().run("calibration measurements", database, parent.path / "reviews")
    second = EvidenceReviewController().run("calibration measurements", database, parent.path / "reviews")
    require(read_json(first / "state.json") == read_json(second / "state.json"), "cross-run review content differs")
    with Library(database) as library:
        new = library.ingest("A revised calibration note withdraws its earlier measurement interpretation.",
                             uri="fixture:study", version="2", title="Calibration note", rights="AIM fixture")
        superseded = audit_review(first, database)
        require(superseded["current_source_status"][sid] == "SUPERSEDED", "old review failed to report supersession")
        library.retire(new, status="RETRACTED", actor="fixture-author", reason="deliberate lifecycle control")
        final = library.audit()
    empty = EvidenceReviewController().run("calibration measurements", database, parent.path / "reviews")
    require(read_json(empty / "summary.json")["selected_excerpts"] == 0, "retired document entered new review")
    return {"initial": initial, "final": final, "superseded_review_audit": superseded,
            "review_runs": [str(p.relative_to(parent.path)) for p in (first, second, empty)],
            "reviews": [audit_review(p, database) for p in (first, second, empty)], "cross_run_state_equal": True}


def crash_rollback(parent):
    database = parent.path / "crash.sqlite"
    with Library(database, create=True) as library:
        before = library.audit()
    script = """import os, sys
from aim.library import Library, identity
text = 'An uncommitted calibration record.'
with Library(sys.argv[1]) as library:
    with library.transaction(write=True):
        library._ingest(text, identity(text, uri='fixture:crash', version='1', title='Crash fixture', rights='AIM fixture'))
        os._exit(23)
"""
    (parent.path / "crash-worker.py").write_text(script)
    result = subprocess.run([sys.executable, "-c", script, str(database)], cwd=ROOT, capture_output=True, text=True, timeout=30)
    (parent.path / "crash-worker.log").write_text(result.stdout + result.stderr)
    require(result.returncode == 23, "crash control did not reach the intended interruption")
    with Library(database) as library:
        after = library.audit()
    require(before == after, "uncommitted crash writes survived recovery")
    return {"exit_code": result.returncode, "before": before, "after": after, "unchanged": True,
            "scope": "one forced process exit before commit; not a power-loss or disk-failure certification"}


def expected_failures(parent):
    from .contracts import ContractError
    database = parent.path / "reject.sqlite"
    with Library(database, create=True):
        pass
    source = parent.path / "rejected-source.md"
    source.write_text("AIM authored calibration fixture.")
    manifest = parent.path / "rejected-manifest.json"
    write_json(manifest, {"schema": "aim-library-intake-v1", "chunk_size": 64, "overlap": 8, "documents": [
        {"path": source.name, "sha256": "0" * 64, "uri": "fixture:rejected", "version": "1", "title": "Fixture",
         "rights": "AIM fixture", "retrieval_allowed": True,
         "privacy_review": {"status": "approved", "reviewer": "fixture-author", "basis": "synthetic"}}]})
    try:
        ingest_manifest(database, manifest, parent.path / "expected-failures")
    except ContractError:
        pass
    else:
        raise ContractError("invalid intake hash unexpectedly accepted")
    class Outsider(ExtractiveResearcher):
        def propose(self, question, hits):
            return [ExcerptProposal("outside-retrieval", "fabricated")]
    try:
        EvidenceReviewController(researcher=Outsider()).run("calibration", database, parent.path / "expected-failures")
    except ContractError:
        pass
    else:
        raise ContractError("unretrieved evidence unexpectedly accepted")
    records = [{"run": str(p.parent.relative_to(parent.path)), **read_json(p)}
               for p in sorted((parent.path / "expected-failures").glob("*/status.json"))]
    require(len(records) == 2 and all(r["status"] == "FAILED" for r in records), "expected failure records missing")
    with Library(database, read_only=True) as library:
        require(library.audit()["versions"] == 0, "rejected operations mutated source inventory")
    return records


def run_study(config, runs, regression=False):
    validate_config(config)
    with Run(Path(runs), "phase-3f-library", {"study": config, "regression": regression}) as run:
        if regression:
            result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                                    cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            (run.path / "tests.log").write_text(result.stdout)
            require(result.returncode == 0 and re.search(r"OK \(skipped=", result.stdout) is None, "regression failed or skipped; log retained")
            print("Full regression passed", flush=True)
        write_json(run.path / "benchmark.json", benchmark(run.path, config))
        print("Library benchmark completed", flush=True)
        write_json(run.path / "lifecycle.json", lifecycle(run))
        write_json(run.path / "crash-recovery.json", crash_rollback(run))
        write_json(run.path / "expected-failures.json", expected_failures(run))
    return run.path


def export_evidence(run, destination):
    run, destination = Path(run), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    names = ["manifest.json", "status.json", "events.jsonl", "benchmark.json", "lifecycle.json", "crash-recovery.json", "expected-failures.json"]
    if (run / "tests.log").exists():
        names.append("tests.log")
    for name in names:
        shutil.copyfile(run / name, destination / name)
    for row in read_json(run / "lifecycle.json")["review_runs"]:
        target = destination / row
        target.mkdir(parents=True)
        for name in ("manifest.json", "status.json", "summary.json", "retrieval.json", "proposals.json", "state.json", "response.md", "replay-audit.json", "library-audit.json"):
            shutil.copyfile(run / row / name, target / name)
    for row in read_json(run / "expected-failures.json"):
        target = destination / row["run"]
        target.mkdir(parents=True)
        for name in ("status.json", "failure.txt", "manifest.json"):
            shutil.copyfile(run / row["run"] / name, target / name)
    files = {str(p.relative_to(destination)): file_hash(p) for p in sorted(destination.rglob("*")) if p.is_file()}
    write_json(destination / "export.json", {"source_run": str(run), "files": files,
               "scope": "portable text evidence; original SQLite snapshots, source archives and memory stores remain in the retained run"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "library-study.json")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    run = run_study(read_json(args.config), args.runs, args.regression)
    if args.export:
        export_evidence(run, args.export)
    print(run)


if __name__ == "__main__":
    main()

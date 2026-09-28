"""Reproduce the Phase 3F application over six existing AIM engineering guides.

Run from model-1 with PYTHONPATH=. and the pinned Python environment.
No scientific relevance scores or training permissions are inferred.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from aim.corpus import read_json, require
from aim.tracking import ROOT, Run, digest, file_hash, write_json

DOCUMENTS = ["TRAINING.md", "JUDGE_SHIFT.md", "HARDWARE_AND_CLUSTER.md", "LANGUAGE_FITTING.md",
             "PRETRAINING.md", "DISTRIBUTED_PRETRAINING.md"]
QUESTIONS = ["RLHF preference optimization RLVR reward calibrated Judge",
             "VIT CPU cluster memory 7B DDP", "checkpoint resume exact tokenizer BPE",
             "calibration OOD uncertainty", "free generation validation checker fitting",
             "photonic superconducting neutrino"]


def main():
    inputs = [Path(__file__), *[ROOT / "docs" / name for name in DOCUMENTS]]
    with Run(ROOT / "runs", "phase-3f-project-documents", {"documents": DOCUMENTS, "questions": QUESTIONS,
             "scope": "application demonstration over existing project guides; no relevance or factuality labels"}, inputs) as run:
        shutil.copyfile(__file__, run.path / "application.py")
        directory = run.path / "documents"
        directory.mkdir()
        rows = []
        for name in DOCUMENTS:
            source = ROOT / "docs" / name
            shutil.copyfile(source, directory / name)
            sha = file_hash(source)
            rows.append({"path": name, "sha256": sha, "uri": "aim:docs/" + name,
                         "version": "sha256:" + sha, "title": source.read_text().splitlines()[0].lstrip("# "),
                         "rights": "Project owner requested use of AIM research documents for local project work; no training or redistribution license inferred",
                         "retrieval_allowed": True,
                         "privacy_review": {"status": "approved", "reviewer": "AIM implementation agent",
                                            "basis": "Read these six project engineering guides for this local application; technical records and reported hardware, no personal datasets or credentials observed; not independent privacy certification"}})
        manifest = directory / "manifest.json"
        write_json(manifest, {"schema": "aim-library-intake-v1", "chunk_size": 1024, "overlap": 128, "documents": rows})
        library = run.path / "library.sqlite"
        commands = []
        def invoke(command, *arguments):
            args = [sys.executable, "-m", "aim.library_cli", command, "--library", str(library),
                    "--runs", str(run.path / "operations"), *map(str, arguments)]
            result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=60)
            log = run.path / f"command-{len(commands):02d}.log"
            log.write_text(result.stdout + result.stderr)
            record = {"command": command, "arguments": list(map(str, arguments)), "exit_code": result.returncode,
                      "log": log.name, "log_sha256": file_hash(log)}
            commands.append(record)
            run.event("CLI_COMPLETED", record)
            require(result.returncode == 0, "application command failed; output retained")
            child = Path(result.stdout.strip().splitlines()[-1])
            require(child.is_relative_to(run.path) and child.is_dir(), "unexpected application run path")
            record["run"] = str(child.relative_to(run.path))
            require(read_json(child / "status.json")["status"] == "COMPLETED", "application child failed")
            return child
        invoke("init")
        intake = invoke("ingest", "--manifest", manifest)
        receipt = read_json(intake / "receipt.json")
        invoke("list")
        invoke("history")
        invoke("source", "--source-id", receipt["source_ids"][0])
        invoke("search", "--question", QUESTIONS[0])
        reviews = []
        for question in QUESTIONS:
            child = invoke("review", "--question", question)
            summary = read_json(child / "summary.json")
            reviews.append({"question": question, "run": str(child.relative_to(run.path)), **summary,
                            "response_sha256": file_hash(child / "response.md"), "audit": read_json(child / "replay-audit.json")})
        replay = invoke("replay", "--review-run", run.path / reviews[0]["run"])
        audit = invoke("audit")
        require(reviews[-1]["selected_excerpts"] == 0, "unmatched vocabulary control retrieved an excerpt")
        write_json(run.path / "summary.json", {"documents": len(rows), "manifest_hash": digest(read_json(manifest)),
                   "commands": commands, "reviews": reviews, "library_audit": read_json(audit / "audit.json"),
                   "live_replay_audit": read_json(replay / "audit.json"), "verified_factual_claims": 0,
                   "quality_metrics": None, "quality_note": "No independent relevance labels; returned counts are not answer-quality measurements"})
    print(run.path)


if __name__ == "__main__":
    main()

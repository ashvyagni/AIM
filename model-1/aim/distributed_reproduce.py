"""Repeatable local DDP build validation, with retained worker-failure drills."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .checkpoint_bundle import load_bundle, scan
from .corpus import Corpus, intake, require
from .corpus_fixture import create_fixture
from .distributed_jobs import DistributedJobError, audit_job, export_initialization, launch, rank_metrics
from .distributed_reference import compare_resume, compare_serial, serial_reference
from .tokenization import ByteTokenizer, fit_bpe
from .tracking import ROOT, Run, file_hash, write_json


def reproduce(runs):
    from .training import train
    config_path = ROOT/"configs/distributed-pretrain-smoke.json"
    config = json.loads(config_path.read_text())
    with Run(Path(runs), "phase-3b-build", {"config": config, "serial_tolerance": 2e-6,
             "scope": "local same-host systems validation; no physical cluster or model quality claim"}, [config_path]) as run:
        tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (run.path/"tests.log").write_text(tests.stdout)
        require(tests.returncode == 0, "full regression failed; retained tests.log")
        print("Full regression passed", flush=True)
        corpus_path = intake(create_fixture(run.path/"fixture"), run.path/"intake")
        corpus = Corpus(corpus_path)
        def record(job):
            summary = json.loads((job/"job/summary.json").read_text())
            return {"path": str(job.relative_to(run.path)), "summary": summary,
                    "status": json.loads((job/"status.json").read_text()), "audit": audit_job(job, corpus_path),
                    "rank_metrics": rank_metrics(job), "checkpoint_scan": scan(job/"job/checkpoints")}
        arms, jobs = {}, {}
        for name, tokenizer in (("byte", ByteTokenizer()), ("bpe", fit_bpe(corpus, 32))):
            write_json(run.path/(name+"-tokenizer.json"), tokenizer.specification())
            full = launch(config, corpus_path, run.path/"jobs", tokenizer=tokenizer)
            half = launch({**config, "steps": 2}, corpus_path, run.path/"jobs", tokenizer=tokenizer)
            resumed = launch(config, corpus_path, run.path/"jobs", tokenizer=tokenizer, resume=half/"job/checkpoints/step-000002")
            reference = serial_reference(config, corpus_path, run.path/"references", tokenizer=tokenizer)
            final = full/"job/checkpoints/step-000006"
            arms[name] = {"full": record(full), "partial": record(half), "resumed": record(resumed),
                          "exact_resume": compare_resume(final, resumed/"job/checkpoints/step-000006"),
                          "serial_reference": {"path": str(reference.relative_to(run.path)),
                                               "comparison": compare_serial(reference, final),
                                               "metrics": json.loads((reference/"metrics.json").read_text())}}
            # Compare actual consumed batches, not merely final cursors.
            oracle = arms[name]["serial_reference"]["metrics"]["updates"]
            for r, metrics in enumerate(arms[name]["full"]["rank_metrics"]):
                require([u["batch_hashes"] for u in metrics["updates"]] == [u["rank_batch_hashes"][r] for u in oracle], "oracle consumed different batches")
            jobs[name] = full
            print(name+": serial agreement and exact two-rank resume passed", flush=True)

        drills = {}
        for mode, restart_step in (("before_publish", 2), ("after_publish", 4)):
            try:
                launch(config, corpus_path, run.path/"jobs", fault={"rank": 1, "step": 4, "mode": mode})
            except DistributedJobError as exc:
                failed = exc.path
            else:
                raise RuntimeError("failure drill unexpectedly completed")
            log = (failed/"launcher.log").read_text()
            require("INJECTED_WORKER_FAILURE" in log, "job failed for an unexpected reason")
            published = failed/"job/checkpoints"/f"step-{restart_step:06d}"
            load_bundle(published)
            listing = scan(failed/"job/checkpoints")
            if mode == "before_publish":
                require(any(r["status"] == "INCOMPLETE" and "000004" in r["name"] for r in listing), "pre-publication failure did not retain partial state")
            recovered = launch(config, corpus_path, run.path/"jobs", resume=published)
            drills[mode] = {"failed_path": str(failed.relative_to(run.path)),
                            "failed_status": json.loads((failed/"status.json").read_text()),
                            "injected_marker_present": True, "launcher_log_sha256": file_hash(failed/"launcher.log"),
                            "checkpoint_scan": listing, "restart_step": restart_step, "recovered": record(recovered),
                            "exact_recovery": compare_resume(jobs["byte"]/"job/checkpoints/step-000006", recovered/"job/checkpoints/step-000006")}
            print(mode+": failed job retained; exact recovery passed", flush=True)
        four = launch({**config, "steps": 2}, corpus_path, run.path/"jobs", world_size=4)
        print("Four local ranks passed replica and data audits", flush=True)
        initialization = export_initialization(jobs["byte"]/"job/checkpoints/step-000006", run.path/"exports")
        sft_config = {k: v for k, v in config.items() if k not in {"accumulation_steps", "collective_timeout_seconds", "checkpoint_every", "eval_batches"}}
        sft_config.update(stage="sft", steps=2)
        sft = train(sft_config, run.path/"sft", initialize=initialization)
        write_json(run.path/"results.json", {"corpus_path": str(corpus_path.relative_to(run.path)), "corpus_hash": corpus.fingerprint,
                   "arms": arms, "failure_drills": drills, "four_rank": record(four),
                   "sft_bridge": {"path": str(sft.relative_to(run.path)), "initialization": str(initialization.relative_to(run.path)),
                                  "checkpoint_sha256": file_hash(sft/"checkpoint.pt"), "metrics": json.loads((sft/"metrics.json").read_text())},
                   "scope": "fixed-topology local Gloo integration; no scientific capability or VIT scaling evidence"})
    return run.path


def export(path, destination):
    path, destination = Path(path), Path(destination)
    require(json.loads((path/"status.json").read_text())["status"] == "COMPLETED", "cannot publish incomplete build as completed")
    results = json.loads((path/"results.json").read_text())
    corpus_path = path/results["corpus_path"]
    all_jobs = [record[stage] for record in results["arms"].values() for stage in ("full", "partial", "resumed")]
    all_jobs += [d["recovered"] for d in results["failure_drills"].values()]+[results["four_rank"]]
    for job in all_jobs:
        require(audit_job(path/job["path"], corpus_path) == job["audit"], "export data audit changed")
    for arm in results["arms"].values():
        full = path/arm["full"]["path"]/"job/checkpoints/step-000006"
        require(compare_resume(full, path/arm["resumed"]["path"]/"job/checkpoints/step-000006") == arm["exact_resume"], "resume export audit changed")
        require(compare_serial(path/arm["serial_reference"]["path"], full) == arm["serial_reference"]["comparison"], "serial export audit changed")
    full = path/results["arms"]["byte"]["full"]["path"]/"job/checkpoints/step-000006"
    for drill in results["failure_drills"].values():
        require(file_hash(path/drill["failed_path"]/"launcher.log") == drill["launcher_log_sha256"], "failed-job log changed")
        require(compare_resume(full, path/drill["recovered"]["path"]/"job/checkpoints/step-000006") == drill["exact_recovery"], "recovery export audit changed")
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("results.json", "tests.log", "byte-tokenizer.json", "bpe-tokenizer.json"):
        shutil.copyfile(path/name, destination/name)
    manifest = json.loads((path/"manifest.json").read_text())
    write_json(destination/"manifest.json", {"run_id": path.name, "git_commit": manifest["environment"]["git_commit"],
               "configuration": manifest["configuration"], "code_hash": manifest["code_hash"], "code_hashes": manifest["code_hashes"],
               "environment": {k: manifest["environment"][k] for k in ("python", "platform", "machine", "logical_cpus", "packages")},
               "status": json.loads((path/"status.json").read_text()), "files": {p.name: file_hash(p) for p in sorted(destination.iterdir())},
               "scope": "portable measured results; worker logs, full rank/checkpoint records and failures remain in the local run"})


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

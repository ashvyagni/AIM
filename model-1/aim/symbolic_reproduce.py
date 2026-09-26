"""Reproduce the build validation; no learned-model promotion or scale decision."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .symbolic_eval import evaluate_symbolic
from .symbolic_judge import CalibratedSymbolicJudge, train_symbolic_judge
from .symbolic_loop import audit_symbolic_run
from .tracking import ROOT, Run, file_hash, write_json


def reproduce(runs):
    from .training import train
    config_paths = [ROOT/"configs"/f"symbolic-{s}.json" for s in ("sft", "preference", "rlvr", "judge")]
    configs = {p.stem.removeprefix("symbolic-"): json.loads(p.read_text()) for p in config_paths}
    with Run(Path(runs), "phase-2c-build", {"configs": configs, "scope": "engineering smoke run; single seed; no promotion"}, config_paths) as run:
        test = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                              cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (run.path/"tests.log").write_text(test.stdout)
        if test.returncode:
            raise RuntimeError("full regression suite failed; retained tests.log")
        print("Full regression completed", flush=True)
        checkpoints, training = {}, {}
        parent = None
        for stage in ("sft", "preference", "rlvr"):
            path = train(configs[stage], run.path/"training", initialize=parent)
            parent = path/"checkpoint.pt"
            checkpoints[stage] = parent
            training[stage] = {"path": str(path.relative_to(run.path)), "checkpoint_sha256": file_hash(parent),
                               "metrics": json.loads((path/"metrics.json").read_text()),
                               "status": json.loads((path/"status.json").read_text())}
            print(stage+" completed", flush=True)
        judge_path = train_symbolic_judge(configs["judge"], run.path/"training")
        training["judge"] = {"path": str(judge_path.relative_to(run.path)),
                             "checkpoint_sha256": file_hash(judge_path/"checkpoint.pt"),
                             "metrics": json.loads((judge_path/"metrics.json").read_text()),
                             "status": json.loads((judge_path/"status.json").read_text())}
        judge = CalibratedSymbolicJudge(judge_path/"checkpoint.pt", cost=.2)
        evaluations = {}
        arms = [("reference", None, None)] + [(s, p, None) for s, p in checkpoints.items()]
        arms += [("reference_with_judge", None, judge), ("rlvr_with_judge", checkpoints["rlvr"], judge)]
        for name, checkpoint, decision in arms:
            path, metrics = evaluate_symbolic(ROOT/"eval/symbolic-v1.json", run.path/"evaluation", checkpoint, decision)
            evaluations[name] = {"path": str(path.relative_to(run.path)), "metrics": metrics}
            print(name+": "+json.dumps(metrics), flush=True)
        write_json(run.path/"results.json", {"training": training, "evaluations": evaluations,
                   "scope": "build validation with exposed engineering fixtures; no causal stage comparison or scale claim"})
    return run.path


def export(path, destination):
    path, destination = Path(path), Path(destination)
    if json.loads((path/"status.json").read_text())["status"] != "COMPLETED":
        raise ValueError("cannot export an incomplete build as completed")
    results = json.loads((path/"results.json").read_text())
    episodes = {}
    for name, record in results["evaluations"].items():
        evaluation = path/record["path"]
        document = json.loads((evaluation/"results.json").read_text())
        for item in document["episodes"]:
            audited = audit_symbolic_run(evaluation/item["path"])
            if audited != item["audit"]:
                raise ValueError("episode audit changed")
        episodes[name] = document
    # Export is an immutable review bundle; full SQLite stores/weights stay local.
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination/"results.json", results)
    write_json(destination/"episodes.json", episodes)
    shutil.copyfile(path/"tests.log", destination/"tests.log")
    manifest = json.loads((path/"manifest.json").read_text())
    write_json(destination/"manifest.json", {"run_id": path.name, "configuration": manifest["configuration"],
               "code_hash": manifest["code_hash"], "code_hashes": manifest["code_hashes"],
               "git_commit": manifest["environment"]["git_commit"],
               "environment": {k: manifest["environment"][k] for k in ("python", "platform", "machine", "logical_cpus", "packages")},
               "status": json.loads((path/"status.json").read_text()),
               "scope": "portable evidence; full checkpoint/source/ledger artifacts retained in local runs",
               "files": {p.name: file_hash(p) for p in sorted(destination.iterdir()) if p.is_file()}})


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

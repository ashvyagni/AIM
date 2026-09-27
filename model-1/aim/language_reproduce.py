"""Deferred Phase 3D audit entry point. Does no work until explicitly invoked."""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .corpus import Corpus, intake, read_json, require
from .corpus_fixture import create_fixture
from .language_contract import read_source
from .tracking import ROOT, Run, file_hash, write_json


def compare(full_path, resumed_path):
    from .checkpoint_bundle import tree_hash
    full, resumed = read_source(full_path), read_source(resumed_path)
    keys = ("model", "optimizer", "reference", "torch_rng", "step", "scored_tokens", "stable_config",
            "dataset_hash", "encoding_manifest_hash", "tokenizer", "encoding", "task", "stage", "output_contract")
    checks = {key: tree_hash(full[key]) == tree_hash(resumed[key]) for key in keys}
    require(all(checks.values()), "language continuation differs from uninterrupted state")
    return {"checks": checks, "exact": True, "scope": "recorded CPU environment; parent lineage legitimately differs"}


def reproduce(runs):
    from .language_artifacts import describe
    from .language_training import train
    from .pretrain import pretrain
    from .tokenization import ByteTokenizer, fit_bpe
    config = read_json(ROOT/"configs/language-sft.json")
    with Run(Path(runs), "phase-3d-audit", {"config": config, "scope": "explicit audit execution; no model promotion"}) as run:
        tested = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (run.path/"tests.log").write_text(tested.stdout)
        require(tested.returncode == 0 and re.search(r"OK \(skipped=", tested.stdout) is None, "regression failed/skipped; retain log and fix before results")
        print("Regression passed", flush=True)
        corpus = Corpus(intake(create_fixture(run.path/"fixture"), run.path/"intake"))
        arms = {}
        for name, tokenizer in (("byte", ByteTokenizer()), ("bpe", fit_bpe(corpus, 32))):
            pcfg = {key: config[key] for key in ("seed", "batch_size", "learning_rate", "weight_decay", "checkpoint_every", "model")}
            pcfg.update(steps=2, eval_batches=2)
            pretrained = pretrain(pcfg, corpus.path, run.path/"pretrain", tokenizer)
            parent = pretrained/"checkpoint.pt"
            stages = {}
            for stage in ("sft", "preference", "rlvr"):
                cfg = {**read_json(ROOT/f"configs/language-{stage}.json"), "steps": 4}
                full = train(cfg, run.path/"training", tokenizer, initialize=parent)
                partial = train({**cfg, "steps": 2}, run.path/"training", tokenizer, initialize=parent)
                resumed = train(cfg, run.path/"training", tokenizer, resume=partial/"checkpoint.pt")
                stages[stage] = {"full": str(full.relative_to(run.path)), "partial": str(partial.relative_to(run.path)),
                                 "resumed": str(resumed.relative_to(run.path)), "description": describe(full/"checkpoint.pt"),
                                 "metrics": read_json(full/"metrics.json"), "continuation": compare(full/"checkpoint.pt", resumed/"checkpoint.pt")}
                parent = full/"checkpoint.pt"
                print(name+" "+stage+": exact continuation comparison passed", flush=True)
            arms[name] = {"pretraining": str(pretrained.relative_to(run.path)), "tokenizer": tokenizer.specification(), "stages": stages}
        from .language_integration import run_integration
        integration = run_integration(run, corpus, arms)
        write_json(run.path/"results.json", {"arms": arms, "integration": integration, "corpus_path": str(corpus.path.relative_to(run.path)),
                   "scope": "small from-scratch vocabulary/objective integration; no cross-tokenizer capability conclusion"})
    return run.path


def export(path, destination):
    from .language_artifacts import describe
    path, destination = Path(path), Path(destination)
    require(read_json(path/"status.json")["status"] == "COMPLETED", "cannot export failed/pending audit as completed")
    results = read_json(path/"results.json")
    from .language_integration import audit_integration
    integration_audit = audit_integration(path, results["integration"], results["arms"], path/results["corpus_path"])
    for arm in results["arms"].values():
        for stage in arm["stages"].values():
            require(compare(path/stage["full"]/"checkpoint.pt", path/stage["resumed"]/"checkpoint.pt") == stage["continuation"], "continuation export changed")
            require(describe(path/stage["full"]/"checkpoint.pt") == stage["description"], "checkpoint export changed")
            require(read_json(path/stage["full"]/"metrics.json") == stage["metrics"], "metrics export changed")
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("results.json", "tests.log"):
        shutil.copyfile(path/name, destination/name)
    manifest = read_json(path/"manifest.json")
    write_json(destination/"manifest.json", {"run_id": path.name, "git_commit": manifest["environment"]["git_commit"],
               "code_hash": manifest["code_hash"], "code_hashes": manifest["code_hashes"], "configuration": manifest["configuration"],
               "environment": {k: manifest["environment"][k] for k in ("python", "platform", "machine", "logical_cpus", "packages")},
               "status": read_json(path/"status.json"), "integration_audit": integration_audit,
               "files": {n: file_hash(destination/n) for n in ("results.json", "tests.log")}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    path = reproduce(args.runs)
    if args.export: export(path, args.export)
    print(path)


if __name__ == "__main__": main()

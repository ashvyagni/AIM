"""Build validation for corpus/tokenizer/pretraining, including a retained failed run."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from .corpus import Corpus, intake
from .corpus_fixture import create_fixture
from .pretrain import pretrain
from .tokenization import ByteTokenizer, fit_bpe, compare_tokenizers
from .tracking import ROOT, Run, file_hash, write_json


def compare_checkpoints(full, resumed):
    import torch
    from .neural import read_checkpoint
    def identical(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and torch.equal(a, b)
        if isinstance(a, dict):
            return isinstance(b, dict) and set(a) == set(b) and all(identical(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return type(a) is type(b) and len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
        return a == b
    a, b = (read_checkpoint(p/"checkpoint.pt") for p in (full, resumed))
    checked = {k: identical(a[k], b[k]) for k in ("model", "optimizer", "torch_rng", "step", "scored_tokens", "cursor", "corpus_hash", "tokenizer", "stable_config")}
    ma, mb = (json.loads((p/"metrics.json").read_text()) for p in (full, resumed))
    checked["validation_metrics"] = ma["final"] == mb["final"]
    start = mb["updates"][0]["step"]-1
    checked["continued_batch_hashes"] = [u["batch_hash"] for u in ma["updates"]][start:] == [u["batch_hash"] for u in mb["updates"]]
    if not all(checked.values()):
        raise RuntimeError("pretraining continuation differs: "+str(checked))
    return checked


def reproduce(runs):
    from .training import train, save_checkpoint
    config_path = ROOT/"configs/pretrain-smoke.json"
    config = json.loads(config_path.read_text())
    with Run(Path(runs), "phase-3a-build", {"pretrain": config, "bpe_merges": 32,
             "scope": "generated-corpus engineering run; no quality or scale promotion"}, [config_path]) as run:
        tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (run.path/"tests.log").write_text(tests.stdout)
        if tests.returncode:
            raise RuntimeError("full regression failed; log retained")
        print("Full regression passed", flush=True)
        manifest = create_fixture(run.path/"fixture")
        corpus_path = intake(manifest, run.path/"intake")
        corpus = Corpus(corpus_path)
        tokenizers = {"byte": ByteTokenizer(), "bpe": fit_bpe(corpus, 32)}
        write_json(run.path/"tokenizer-comparison.json", compare_tokenizers(corpus, tokenizers))
        outcomes, full_paths = {}, {}
        def record(path):
            return {"path": str(path.relative_to(run.path)), "checkpoint_sha256": file_hash(path/"checkpoint.pt"),
                    "metrics": json.loads((path/"metrics.json").read_text()), "status": json.loads((path/"status.json").read_text())}
        for name, tokenizer in tokenizers.items():
            write_json(run.path/(name+"-tokenizer.json"), tokenizer.specification())
            full = pretrain(config, corpus_path, run.path/"training", tokenizer)
            half = pretrain({**config, "steps": 6}, corpus_path, run.path/"training", tokenizer)
            resumed = pretrain(config, corpus_path, run.path/"training", tokenizer, half/"checkpoint.pt")
            full_paths[name] = full
            outcomes[name] = {"full": record(full), "partial": record(half), "resumed": record(resumed),
                              "exact_resume": compare_checkpoints(full, resumed)}
            print(name+" pretraining/resume passed", flush=True)

        class InjectedInterruption(RuntimeError):
            pass
        failed_path = []
        def interrupt_after_publication(child, state, filename):
            save_checkpoint(child, state, filename)
            failed_path.append(child.path)
            raise InjectedInterruption("Deliberate recovery drill: interrupt immediately after durable checkpoint publication")
        try:
            with patch("aim.training.save_checkpoint", side_effect=interrupt_after_publication):
                pretrain(config, corpus_path, run.path/"interruption-drill")
        except InjectedInterruption:
            pass
        if len(failed_path) != 1:
            raise RuntimeError("recovery drill did not reach its intended interruption")
        failed = failed_path[0]
        checkpoint = failed/"checkpoint-step-000004.pt"
        recovered = pretrain(config, corpus_path, run.path/"training", resume=checkpoint)
        recovery = {"failed_path": str(failed.relative_to(run.path)), "failed_status": json.loads((failed/"status.json").read_text()),
                    "checkpoint_sha256": file_hash(checkpoint), "recovered": record(recovered),
                    "exact_recovery": compare_checkpoints(full_paths["byte"], recovered),
                    "scope": "application-level injected exception; not an OS kill, power loss or distributed fault"}
        sft_config = {k: v for k, v in config.items() if k not in {"checkpoint_every", "eval_batches"}}
        sft_config.update(stage="sft", steps=2)
        sft = train(sft_config, run.path/"sft", initialize=full_paths["byte"]/"checkpoint.pt")
        write_json(run.path/"results.json", {"corpus": {"path": str(corpus_path.relative_to(run.path)),
                   "hash": corpus.fingerprint, "documents": len(corpus.index["documents"]), "bytes": corpus.index["total_bytes"],
                   "splits": {s: len(corpus.records(s)) for s in ("train", "validation", "test")}},
                   "pretraining": outcomes, "recovery": recovery, "sft_bridge": record(sft),
                   "scope": "native next-token infrastructure on constructed text; no research-capability result"})
    return run.path


def export(path, destination):
    path, destination = Path(path), Path(destination)
    results = json.loads((path/"results.json").read_text())
    if json.loads((path/"status.json").read_text())["status"] != "COMPLETED":
        raise RuntimeError("build did not complete")
    for record in results["pretraining"].values():
        compare_checkpoints(path/record["full"]["path"], path/record["resumed"]["path"])
    compare_checkpoints(path/results["pretraining"]["byte"]["full"]["path"], path/results["recovery"]["recovered"]["path"])
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("results.json", "tests.log", "tokenizer-comparison.json", "byte-tokenizer.json", "bpe-tokenizer.json"):
        shutil.copyfile(path/name, destination/name)
    failure = (path/results["recovery"]["failed_path"]/"failure.txt").read_text()
    (destination/"injected-failure.txt").write_text(failure.replace(str(ROOT), "model-1"))
    corpus = Corpus(path/results["corpus"]["path"])
    # Review artifacts contain only this project-generated fixture. They are not
    # a generic exporter for external/private corpus text or source paths.
    fixture = [{"id": row["id"], "split": row["split"], "type": row["content_type"], "sha256": row["sha256"], "text": text}
               for split in ("train", "validation", "test") for row, text in corpus.documents(split)]
    write_json(destination/"fixture.json", fixture)
    manifest = json.loads((path/"manifest.json").read_text())
    write_json(destination/"manifest.json", {"run_id": path.name, "code_hash": manifest["code_hash"], "code_hashes": manifest["code_hashes"],
               "git_commit": manifest["environment"]["git_commit"], "configuration": manifest["configuration"],
               "environment": {k: manifest["environment"][k] for k in ("python", "platform", "machine", "logical_cpus", "packages")},
               "status": json.loads((path/"status.json").read_text()),
               "files": {p.name: file_hash(p) for p in sorted(destination.iterdir())},
               "scope": "constructed-text review evidence; full corpora, source snapshots and checkpoints remain in local runs"})


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

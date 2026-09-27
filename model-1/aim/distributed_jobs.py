"""Local launch, preflight and data/checkpoint audits for corpus DDP training."""
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import subprocess
import sys

from .checkpoint_bundle import load_bundle, scan
from .corpus import Corpus, read_json, require
from .distributed_pretrain import validate_inputs
from .sharded_data import RankCorpus, audit_partition, partition
from .tokenization import ByteTokenizer, tokenizer_from_spec
from .token_stream import TokenStream
from .tracking import ROOT, Run, digest, write_json


class DistributedJobError(RuntimeError):
    def __init__(self, path, message):
        self.path = Path(path)
        super().__init__(message+"; retained job: "+str(path))


def preflight(config, corpus_path, tokenizer=None, world_size=2):
    corpus = Corpus(corpus_path)
    tokenizer = tokenizer or ByteTokenizer()
    cfg, layout = validate_inputs(config, corpus, tokenizer, world_size)
    # Check all training/validation objects before starting workers; never test text.
    for split in ("train", "validation"):
        for _ in corpus.documents(split):
            pass
    return {"partition": layout, "ownership_audit": audit_partition(layout, corpus), "parameter_count": cfg.parameter_estimate(),
            "fp32_adam_parameter_gradient_moment_floor_bytes_per_rank": 16*cfg.parameter_estimate(),
            "disk_free_bytes_on_local_workspace": shutil.disk_usage(ROOT).free,
            "effective_sequences_per_update": world_size*config["batch_size"]*config["accumulation_steps"],
            "position_budget_per_update": world_size*config["batch_size"]*config["accumulation_steps"]*cfg.context,
            "scope": "local object integrity and analytical memory floor; excludes activations/runtime; no physical cluster validation"}


def launch(config, corpus_path, runs, world_size=2, tokenizer=None, resume=None, fault=None, deadline=180):
    require(world_size in (1, 2, 4) and type(world_size) is int, "local launcher supports 1, 2 or 4 ranks")
    require(type(deadline) is int and 15 <= deadline <= 600, "local deadline must be 15..600 seconds")
    if fault:
        require(isinstance(fault, dict) and set(fault) == {"rank", "step", "mode"} and type(fault["rank"]) is int and
                0 <= fault["rank"] < world_size and type(fault["step"]) is int and 0 < fault["step"] <= config["steps"] and
                (fault["step"] % config["checkpoint_every"] == 0 or fault["step"] == config["steps"]) and
                fault["mode"] in {"before_publish", "after_publish"}, "fault must name an actual rank/checkpoint boundary")
    tokenizer = tokenizer or ByteTokenizer()
    with Run(Path(runs), "distributed-pretrain-launch", {"config": config, "world_size": world_size,
             "tokenizer": tokenizer.specification(), "resume": str(resume) if resume else None,
             "fault": fault, "deadline_seconds": deadline, "scope": "loopback local processes only"}, [Path(corpus_path)]) as run:
        write_json(run.path/"preflight.json", preflight(config, corpus_path, tokenizer, world_size))
        write_json(run.path/"config.json", config)
        write_json(run.path/"tokenizer.json", tokenizer.specification())
        job = run.path/"job"
        job.mkdir()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", f"--nproc-per-node={world_size}",
                   "--master-addr=127.0.0.1", f"--master-port={port}", "--rdzv-conf=timeout=30", "--max-restarts=0",
                   "-m", "aim.distributed_pretrain", "--config", str((run.path/"config.json").resolve()),
                   "--corpus", str(Path(corpus_path).resolve()), "--job", str(job.resolve()),
                   "--tokenizer", str((run.path/"tokenizer.json").resolve())]
        if resume:
            command += ["--resume", str(Path(resume).resolve())]
        if fault:
            command += ["--fault-rank", str(fault["rank"]), "--fault-step", str(fault["step"]), "--fault-mode", fault["mode"]]
        overrides = {"OMP_NUM_THREADS": "1", "GLOO_SOCKET_IFNAME": "lo0" if platform.system() == "Darwin" else "lo"}
        write_json(run.path/"launch.json", {"command": command, "environment_overrides": overrides})
        with (run.path/"launcher.log").open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **overrides}, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=deadline)
                if code:
                    raise DistributedJobError(run.path, f"worker job exited {code}")
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        write_json(run.path/"data-audit.json", audit_job(run.path, corpus_path))
    return run.path


def rank_metrics(path):
    job = Path(path)/"job"
    records = [read_json(p) for p in sorted((job/"ranks").glob("*/metrics.json"))]
    records.sort(key=lambda r: r["rank"])
    return records


def audit_checkpoint(path, corpus_path):
    common, states, manifest = load_bundle(path)
    corpus = Corpus(corpus_path)
    tokenizer = tokenizer_from_spec(common["tokenizer"])
    layout = partition(corpus, common["world_size"])
    require(common["corpus_hash"] == corpus.fingerprint and common["partition_hash"] == layout["partition_hash"], "checkpoint belongs to another corpus partition")
    for rank, state in enumerate(states):
        TokenStream(RankCorpus(corpus, rank, common["world_size"]), tokenizer, cursor=state["cursor"])
    return {**manifest["audit"], "bundle_hash": manifest["bundle_hash"], "cursor_bindings_valid": True}


def audit_job(path, corpus_path):
    path = Path(path)
    config = read_json(path/"config.json")
    tokenizer = tokenizer_from_spec(read_json(path/"tokenizer.json"))
    corpus = Corpus(corpus_path)
    summary = read_json(path/"job/summary.json")
    layout = read_json(path/"job/partition.json")
    ownership = audit_partition(layout, corpus)
    records = rank_metrics(path)
    world = summary["world_size"]
    require([r["rank"] for r in records] == list(range(world)), "missing or duplicate rank metrics")
    step_rows = {}
    for record in records:
        stream = TokenStream(RankCorpus(corpus, record["rank"], world), tokenizer, cursor=record["start_cursor"])
        require([u["step"] for u in record["updates"]] == list(range(record["start_step"]+1, config["steps"]+1)), "rank skipped or repeated an optimizer step")
        for update in record["updates"]:
            batches = [stream.batch(config["batch_size"], config["model"]["context"]) for _ in range(config["accumulation_steps"])]
            require(update["batch_hashes"] == [digest(b) for b in batches] and update["cursor"] == stream.cursor(), "rank batch/cursor replay mismatch")
            count = sum(sum(sum(row) for row in b["mask"]) for b in batches)
            require(count == update["local_tokens"], "rank token accounting mismatch")
            step_rows.setdefault(update["step"], []).append(update)
    for rows in step_rows.values():
        require(len(rows) == world and all(r["global_tokens"] == sum(x["local_tokens"] for x in rows) for r in rows), "global step token denominator mismatch")
        require(len({r["global_loss"] for r in rows}) == 1, "ranks reported different reduced loss")
    checkpoint = audit_checkpoint(path/"job"/summary["final_checkpoint"], corpus_path)
    common, states, _ = load_bundle(path/"job"/summary["final_checkpoint"])
    require(common["stable_config"] == {k: v for k, v in config.items() if k != "steps"} and common["step"] == config["steps"],
            "checkpoint/job configuration differs")
    for record, state in zip(records, states):
        require(record["local_tokens_total"] == state["local_tokens"] and record["global_tokens_total"] == common["global_tokens"] and
                record["updates"][-1]["cursor"] == state["cursor"], "rank trace totals/cursor differ from checkpoint")
    require(checkpoint["global_tokens"] == summary["global_tokens_total"] == sum(r["local_tokens_total"] for r in records), "final token totals disagree")
    return {"ownership": ownership, "replayed_rank_updates": sum(len(r["updates"]) for r in records),
            "checkpoint": checkpoint, "scope": "batch/cursor/count/certificate replay; no independent proof of optimizer math"}


def export_initialization(bundle, runs):
    from .training import save_checkpoint
    common, _, manifest = load_bundle(bundle)
    with Run(Path(runs), "distributed-initialization-export", {"bundle_hash": manifest["bundle_hash"], "scope": "weights for SFT initialization; not a resumable distributed state"}, [Path(bundle)/"bundle.json"]) as run:
        save_checkpoint(run, {"kind": "pretraining_initialization", "schema": "aim-pretraining-initialization-v1", "origin": "aim-random-init-v1",
                              "model": common["model"], "model_config": common["model_config"], "tokenizer": common["tokenizer"],
                              "parent_bundle_hash": manifest["bundle_hash"], "pretraining_steps": common["step"]}, "initialization.pt")
    return run.path/"initialization.pt"

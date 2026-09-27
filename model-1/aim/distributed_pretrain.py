"""CPU/Gloo corpus pretraining with global-token loss and coordinated checkpoints."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import datetime
import json
import os
from pathlib import Path
import platform
import time

from .checkpoint_bundle import SCHEMA, load_bundle, prepare, publish, save_record, tree_hash
from .corpus import Corpus, read_json, require
from .pretrain import evaluate, validate_config
from .sharded_data import RankCorpus, audit_partition, partition
from .tokenization import ByteTokenizer, tokenizer_from_spec
from .token_stream import TokenStream
from .tracking import Run, digest, write_json


def base_config(config):
    require(isinstance(config, dict) and {"accumulation_steps", "collective_timeout_seconds"} <= set(config), "distributed controls required")
    require(type(config["accumulation_steps"]) is int and 1 <= config["accumulation_steps"] <= 8, "accumulation must be 1..8")
    require(type(config["collective_timeout_seconds"]) is int and 5 <= config["collective_timeout_seconds"] <= 120, "collective timeout must be 5..120 seconds")
    return {k: v for k, v in config.items() if k not in {"accumulation_steps", "collective_timeout_seconds"}}


def validate_inputs(config, corpus, tokenizer, world):
    cfg = validate_config(base_config(config), tokenizer)
    layout = partition(corpus, world)
    audit_partition(layout, corpus)
    require(bool(corpus.records("validation")), "validation split required")
    spec = tokenizer.specification()
    if spec.get("learned"):
        require(spec["provenance"]["corpus_hash"] == corpus.fingerprint and
                spec["provenance"]["train_hashes"] == [r["sha256"] for r in corpus.records("train")], "tokenizer fitting partition differs from corpus")
    return cfg, layout


def scaled_loss(nll_sum, world_size, global_tokens):
    require(type(world_size) is int and world_size > 0 and type(global_tokens) is int and global_tokens > 0, "invalid global-token loss denominator")
    # DDP averages rank gradients. This factor yields the gradient of the
    # single global token mean, even with unequal masks and accumulation.
    return nll_sum * (world_size/global_tokens)


def resume_state(common, rank_states, config, corpus, tokenizer, layout, rank):
    stable = {k: v for k, v in config.items() if k != "steps"}
    require(common["world_size"] == layout["world_size"], "world-size changes require an explicit resharding protocol")
    require(common["stable_config"] == stable and common["corpus_hash"] == corpus.fingerprint and
            common["partition_hash"] == layout["partition_hash"] and common["tokenizer"] == tokenizer.specification(),
            "distributed resume configuration/corpus/partition/tokenizer mismatch")
    require(common["step"] < config["steps"], "resume must advance total optimizer steps")
    state = rank_states[rank]
    stream = TokenStream(RankCorpus(corpus, rank, layout["world_size"]), tokenizer, cursor=state["cursor"])
    return state, stream


def worker(config_path, corpus_path, job, tokenizer_path=None, resume=None, fault=None):
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from .neural import CausalLM
    from .training import seed_all
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    job = Path(job)
    config = read_json(config_path)
    tokenizer = tokenizer_from_spec(read_json(tokenizer_path)) if tokenizer_path else ByteTokenizer()
    inputs = [Path(config_path), Path(corpus_path)]+([Path(tokenizer_path)] if tokenizer_path else [])
    if resume:
        inputs.append(Path(resume)/"bundle.json")
    with Run(job/"ranks", f"ddp-pretrain-rank-{rank}", {"config": config, "rank": rank, "world_size": world,
             "tokenizer": tokenizer.specification(), "resume": str(resume) if resume else None, "fault": fault}, inputs) as run:
        require(type(rank) is int and 0 <= rank < world, "invalid process rank")
        corpus = Corpus(corpus_path)
        cfg, layout = validate_inputs(config, corpus, tokenizer, world)
        seed_all(config["seed"])
        dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=config["collective_timeout_seconds"]))
        try:
            source_hash = read_json(run.path/"manifest.json")["code_hash"]
            identity = {"config": digest(config), "corpus": corpus.fingerprint, "partition": layout["partition_hash"],
                        "tokenizer": digest(tokenizer.specification()), "source": source_hash,
                        "torch": torch.__version__, "python": platform.python_version()}
            peers = [None]*world
            dist.all_gather_object(peers, identity)
            require(all(p == identity for p in peers), "rank source/environment/data/configuration disagreement")
            if rank == 0:
                write_json(job/"partition.json", layout)
                write_json(job/"peer-contract.json", {"identity": identity, "ranks_agree": True, "world_size": world})
            model = CausalLM(cfg)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
            start, local_tokens, global_tokens = 0, 0, 0
            stream = TokenStream(RankCorpus(corpus, rank, world), tokenizer)
            saved_rng, parent_hash = None, None
            if resume:
                common, states, manifest = load_bundle(resume)
                state, stream = resume_state(common, states, config, corpus, tokenizer, layout, rank)
                require(common["model_config"] == asdict(cfg), "resume architecture mismatch")
                model.load_state_dict(common["model"])
                optimizer.load_state_dict(common["optimizer"])
                start, local_tokens, global_tokens = common["step"], state["local_tokens"], common["global_tokens"]
                saved_rng, parent_hash = state["torch_rng"], manifest["bundle_hash"]
            ddp = DDP(model)
            if saved_rng is not None:
                torch.set_rng_state(saved_rng)
            start_cursor = stream.cursor()

            def validation():
                value = [evaluate(model, corpus, tokenizer, base_config(config)) if rank == 0 else None]
                dist.broadcast_object_list(value, src=0)
                ddp.train()
                return value[0]

            initial = validation()
            write_json(run.path/"initial-metrics.json", initial)
            stable = {k: v for k, v in config.items() if k != "steps"}
            shared = {"schema": SCHEMA, "origin": "aim-random-init-v1", "world_size": world,
                      "corpus_hash": corpus.fingerprint, "partition_hash": layout["partition_hash"],
                      "tokenizer_hash": digest(tokenizer.specification()), "config_hash": digest(stable)}

            def checkpoint(step):
                pending = job/"checkpoints"/f".pending-step-{step:06d}"
                if rank == 0:
                    prepare(job/"checkpoints", step)
                dist.barrier()
                state = {**shared, "kind": "distributed_rank", "rank": rank, "step": step,
                         "global_tokens": global_tokens, "local_tokens": local_tokens, "cursor": stream.cursor(),
                         "torch_rng": torch.get_rng_state(), "model_hash": tree_hash(model.state_dict()),
                         "optimizer_hash": tree_hash(optimizer.state_dict())}
                save_record(pending/f"rank-{rank:03d}.pt", state)
                if fault and fault == {"rank": rank, "step": step, "mode": "before_publish"}:
                    raise RuntimeError("INJECTED_WORKER_FAILURE before checkpoint publication")
                dist.barrier()
                if rank == 0:
                    common = {**shared, "kind": "distributed_pretraining_lm", "step": step, "global_tokens": global_tokens,
                              "model_config": asdict(cfg), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                              "stable_config": stable, "tokenizer": tokenizer.specification(), "parent_bundle_hash": parent_hash}
                    publish(pending, common)
                dist.barrier()
                run.event("CHECKPOINT_PUBLISHED", {"step": step})
                if fault and fault == {"rank": rank, "step": step, "mode": "after_publish"}:
                    raise RuntimeError("INJECTED_WORKER_FAILURE after checkpoint publication")

            updates = []
            for step in range(start, config["steps"]):
                began = time.perf_counter()
                batches = [stream.batch(config["batch_size"], cfg.context) for _ in range(config["accumulation_steps"])]
                local_count = sum(sum(sum(row) for row in b["mask"]) for b in batches)
                count = torch.tensor(local_count, dtype=torch.int64)
                dist.all_reduce(count)
                total = int(count)
                require(total > 0, "empty distributed token batch")
                optimizer.zero_grad()
                nll_sum = 0.
                for i, batch in enumerate(batches):
                    context = ddp.no_sync() if i+1 < len(batches) else nullcontext()
                    with context:
                        x, y, mask = (torch.tensor(batch[k]) for k in ("inputs", "targets", "mask"))
                        losses = F.cross_entropy(ddp(x).flatten(0, 1), y.flatten(), reduction="none").reshape_as(mask)
                        local_sum = (losses*mask).sum()
                        require(bool(torch.isfinite(local_sum)), "nonfinite distributed loss")
                        scaled_loss(local_sum, world, total).backward()
                        nll_sum += float(local_sum.detach())
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                aggregate = torch.tensor(nll_sum, dtype=torch.float64)
                dist.all_reduce(aggregate)
                elapsed = torch.tensor(time.perf_counter()-began, dtype=torch.float64)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                local_tokens += local_count
                global_tokens += total
                record = {"step": step+1, "global_loss": float(aggregate)/total, "local_tokens": local_count,
                          "global_tokens": total, "gradient_norm": float(norm), "max_rank_update_seconds": float(elapsed),
                          "batch_hashes": [digest(b) for b in batches], "cursor": stream.cursor()}
                updates.append(record)
                run.metric(step+1, **{k: v for k, v in record.items() if k != "step"})
                if (step+1) % config["checkpoint_every"] == 0 or step+1 == config["steps"]:
                    checkpoint(step+1)
            final = validation()
            replica = {"rank": rank, "host_hash": digest(platform.node()), "model_hash": tree_hash(model.state_dict()),
                       "optimizer_hash": tree_hash(optimizer.state_dict()), "local_tokens": local_tokens}
            replicas = [None]*world
            dist.all_gather_object(replicas, replica)
            require(len({p["model_hash"] for p in replicas}) == len({p["optimizer_hash"] for p in replicas}) == 1,
                    "final replicas disagree")
            result = {"rank": rank, "world_size": world, "start_step": start, "start_cursor": start_cursor,
                      "initial": initial, "final": final, "updates": updates, "global_tokens_total": global_tokens,
                      "local_tokens_total": local_tokens, "replicas": replicas, "parameters": cfg.parameter_estimate(),
                      "scope": "same-host processes" if len({p["host_hash"] for p in replicas}) == 1 else "multiple reported hosts; performance not certified"}
            write_json(run.path/"metrics.json", result)
            if rank == 0:
                write_json(job/"summary.json", {"world_size": world, "steps": config["steps"], "parameters": cfg.parameter_estimate(),
                           "global_tokens_total": global_tokens, "final": final, "replicas_equal": True, "scope": result["scope"],
                           "final_checkpoint": f'checkpoints/step-{config["steps"]:06d}'})
            dist.barrier()
        finally:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--fault-rank", type=int)
    parser.add_argument("--fault-step", type=int)
    parser.add_argument("--fault-mode", choices=("before_publish", "after_publish"))
    args = parser.parse_args()
    fault = None if args.fault_mode is None else {"rank": args.fault_rank, "step": args.fault_step, "mode": args.fault_mode}
    worker(args.config, args.corpus, args.job, args.tokenizer, args.resume, fault)


if __name__ == "__main__":
    main()

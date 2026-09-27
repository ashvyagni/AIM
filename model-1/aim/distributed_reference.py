"""Serial global-batch oracle using the same declared rank streams, without DDP."""
from pathlib import Path
import torch
from torch.nn import functional as F

from .checkpoint_bundle import load_bundle, tree_hash
from .corpus import Corpus, require
from .distributed_pretrain import base_config, validate_inputs
from .neural import CausalLM
from .pretrain import evaluate
from .sharded_data import RankCorpus
from .token_stream import TokenStream
from .tokenization import ByteTokenizer
from .tracking import Run, digest, write_json
from .training import seed_all, save_checkpoint


def serial_reference(config, corpus_path, runs, world_size=2, tokenizer=None):
    tokenizer = tokenizer or ByteTokenizer()
    with Run(Path(runs), "serial-global-batch-reference", {"config": config, "world_size": world_size,
             "tokenizer": tokenizer.specification(), "scope": "same rank-owned batches without DDP; correctness oracle"}, [Path(corpus_path)]) as run:
        corpus = Corpus(corpus_path)
        cfg, layout = validate_inputs(config, corpus, tokenizer, world_size)
        seed_all(config["seed"])
        model = CausalLM(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        streams = [TokenStream(RankCorpus(corpus, r, world_size), tokenizer) for r in range(world_size)]
        updates = []
        for step in range(config["steps"]):
            batches = [[stream.batch(config["batch_size"], cfg.context) for _ in range(config["accumulation_steps"])] for stream in streams]
            count = sum(sum(sum(sum(row) for row in b["mask"]) for b in rows) for rows in batches)
            optimizer.zero_grad()
            value = 0.
            for rank_batches in batches:
                for batch in rank_batches:
                    x, y, mask = (torch.tensor(batch[k]) for k in ("inputs", "targets", "mask"))
                    loss_sum = (F.cross_entropy(model(x).flatten(0,1), y.flatten(), reduction="none").reshape_as(mask)*mask).sum()
                    (loss_sum/count).backward()
                    value += float(loss_sum.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            updates.append({"step": step+1, "global_tokens": count, "global_loss": value/count,
                            "gradient_norm": float(norm), "rank_batch_hashes": [[digest(b) for b in rows] for rows in batches]})
        final = evaluate(model, corpus, tokenizer, base_config(config))
        save_checkpoint(run, {"kind": "serial_ddp_oracle", "origin": "aim-random-init-v1", "model": model.state_dict(),
                              "optimizer": optimizer.state_dict(), "tokenizer": tokenizer.specification(), "partition": layout,
                              "cursors": [s.cursor() for s in streams]}, "reference.pt")
        write_json(run.path/"metrics.json", {"updates": updates, "final": final, "model_hash": tree_hash(model.state_dict())})
    return run.path


def compare_resume(full_bundle, continued_bundle):
    a, ar, _ = load_bundle(full_bundle)
    b, br, _ = load_bundle(continued_bundle)
    common = {k: tree_hash(a[k]) == tree_hash(b[k]) for k in
              ("model", "optimizer", "step", "global_tokens", "model_config", "stable_config", "corpus_hash", "partition_hash", "tokenizer")}
    require(len(ar) == len(br), "resume rank count differs")
    ranks = [{k: tree_hash(x[k]) == tree_hash(y[k]) for k in ("rank", "local_tokens", "global_tokens", "cursor", "torch_rng", "model_hash", "optimizer_hash")}
             for x, y in zip(ar, br)]
    require(all(common.values()) and all(all(r.values()) for r in ranks), "distributed continuation is not exact")
    return {"common": common, "ranks": ranks, "exact": True}


def compare_serial(reference, bundle, tolerance=2e-6):
    from .neural import read_checkpoint
    source = read_checkpoint(Path(reference)/"reference.pt")
    common, ranks, _ = load_bundle(bundle)
    require(source["partition"]["partition_hash"] == common["partition_hash"] and source["tokenizer"] == common["tokenizer"], "oracle/checkpoint data mismatch")
    differences = {k: float((v-common["model"][k]).abs().max()) for k, v in source["model"].items()}
    maximum = max(differences.values())
    require(maximum <= tolerance, "distributed weights disagree with serial global-token oracle")
    require(source["cursors"] == [r["cursor"] for r in ranks], "serial/distributed stream cursors differ")
    return {"max_absolute_parameter_difference": maximum, "tolerance": tolerance, "cursors_equal": True,
            "scope": "numerical agreement with independent serial reduction order; not bitwise equivalence"}

"""Small CPU next-token pretraining with resumable corpus/tokenizer/cursor identity."""
from dataclasses import asdict
from pathlib import Path
import time

from .corpus import Corpus, require
from .tokenization import ByteTokenizer, tokenizer_from_spec
from .token_stream import TokenStream
from .tracking import Run, digest, file_hash, write_json


def validate_config(config, tokenizer):
    from .neural import ModelConfig
    from .training import validate_config as validate_training
    require(isinstance(config, dict) and set(config) == {"seed", "steps", "batch_size", "learning_rate", "weight_decay", "checkpoint_every", "eval_batches", "model"}, "invalid pretraining configuration fields")
    validate_training({**config, "stage": "sft"})
    require(config["steps"] <= 1000 and config["batch_size"] <= 16, "pretraining smoke budget exceeded")
    require(type(config["checkpoint_every"]) is int and 1 <= config["checkpoint_every"] <= 1000, "invalid checkpoint interval")
    require(type(config["eval_batches"]) is int and 1 <= config["eval_batches"] <= 32, "invalid validation batch count")
    require(isinstance(config["model"], dict) and "vocab_size" not in config["model"], "pretraining vocabulary derives from the recorded tokenizer")
    cfg = ModelConfig(**config["model"], vocab_size=tokenizer.vocab_size)
    require(cfg.parameter_estimate() <= 2_000_000 and cfg.context <= 512, "pretraining model exceeds miniature allocation guard")
    return cfg


def evaluate(model, corpus, tokenizer, config):
    import torch
    from torch.nn import functional as F
    stream = TokenStream(corpus, tokenizer, "validation", repeat=False)
    total_loss, total_tokens, batches = 0., 0, 0
    model.eval()
    with torch.no_grad():
        for _ in range(config["eval_batches"]):
            try:
                batch = stream.batch(config["batch_size"], model.cfg.context)
            except StopIteration:
                break
            x, y, mask = (torch.tensor(batch[k]) for k in ("inputs", "targets", "mask"))
            losses = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="none").reshape_as(mask)
            total_loss += float((losses*mask).sum())
            total_tokens += int(mask.sum())
            batches += 1
    require(total_tokens > 0, "no scored validation tokens")
    return {"token_nll": total_loss/total_tokens, "scored_tokens": total_tokens, "batches": batches,
            "scope": "fixed validation prefix; token NLL is not comparable between different tokenizers"}


def pretrain(config, corpus_path, runs, tokenizer=None, resume=None):
    import torch
    from torch.nn import functional as F
    from .neural import CausalLM, read_checkpoint
    from .training import seed_all, save_checkpoint
    tokenizer = tokenizer or ByteTokenizer()
    # Validate even user-constructed tokenizer objects through the serialized contract.
    tokenizer = tokenizer_from_spec(tokenizer.specification())
    inputs = [Path(corpus_path)]+([Path(resume)] if resume else [])
    with Run(Path(runs), "pretrain", {**config, "tokenizer": tokenizer.specification(),
             "resume": str(resume) if resume else None}, inputs) as run:
        cfg = validate_config(config, tokenizer)
        corpus = Corpus(corpus_path)
        require(corpus.records("train") and corpus.records("validation"), "pretraining needs train and validation splits")
        if tokenizer.specification().get("learned"):
            provenance = tokenizer.specification()["provenance"]
            require(provenance["corpus_hash"] == corpus.fingerprint and provenance["train_hashes"] == [r["sha256"] for r in corpus.records("train")],
                    "BPE fitting corpus/train partition does not match pretraining corpus")
        seed_all(config["seed"])
        model = CausalLM(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        stable = {k: v for k, v in config.items() if k != "steps"}
        start, scored_tokens, cursor = 0, 0, None
        if resume:
            source = read_checkpoint(resume)
            require(source.get("kind") == "pretraining_lm" and source.get("schema") == "aim-pretraining-v1", "not an AIM pretraining checkpoint")
            require(source["stable_config"] == stable and source["corpus_hash"] == corpus.fingerprint and
                    source["tokenizer"] == tokenizer.specification() and source["model_config"] == asdict(cfg), "pretraining resume configuration/data/tokenizer mismatch")
            require(type(source["step"]) is int and 0 < source["step"] < config["steps"], "resume must advance total steps")
            model.load_state_dict(source["model"])
            optimizer.load_state_dict(source["optimizer"])
            torch.set_rng_state(source["torch_rng"])
            start, scored_tokens, cursor = source["step"], source["scored_tokens"], source["cursor"]
        stream = TokenStream(corpus, tokenizer, cursor=cursor)
        write_json(run.path/"tokenizer.json", tokenizer.specification())
        write_json(run.path/"corpus-reference.json", {"corpus_hash": corpus.fingerprint, "index": corpus.index,
                   "scope": "content/rights metadata; actual corpus objects retained in intake artifact"})
        initial = evaluate(model, corpus, tokenizer, config)
        write_json(run.path/"initial-metrics.json", initial)
        updates, update_seconds = [], 0.

        def checkpoint(filename, step):
            save_checkpoint(run, {"kind": "pretraining_lm", "schema": "aim-pretraining-v1", "origin": "aim-random-init-v1",
                                  "model_config": asdict(cfg), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                  "torch_rng": torch.get_rng_state(), "step": step, "scored_tokens": scored_tokens,
                                  "cursor": stream.cursor(), "stable_config": stable, "corpus_hash": corpus.fingerprint,
                                  "tokenizer": tokenizer.specification(), "parent_sha256": file_hash(resume) if resume else None}, filename)

        model.train()
        for step in range(start, config["steps"]):
            began = time.perf_counter()
            batch = stream.batch(config["batch_size"], cfg.context)
            x, y, mask = (torch.tensor(batch[k]) for k in ("inputs", "targets", "mask"))
            optimizer.zero_grad()
            losses = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="none").reshape_as(mask)
            count = int(mask.sum())
            require(count > 0, "pretraining batch has no targets")
            loss = (losses*mask).sum()/count
            require(bool(torch.isfinite(loss)), "nonfinite pretraining loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            elapsed = time.perf_counter()-began
            update_seconds += elapsed
            scored_tokens += count
            metric = {"step": step+1, "loss": float(loss.detach()), "gradient_norm": float(norm), "scored_tokens": count,
                      "batch_hash": digest(batch), "cursor": stream.cursor(), "update_seconds": elapsed}
            updates.append(metric)
            run.metric(step+1, **{k: v for k, v in metric.items() if k != "step"})
            if (step+1) % config["checkpoint_every"] == 0 and step+1 != config["steps"]:
                checkpoint(f"checkpoint-step-{step+1:06d}.pt", step+1)
        final = evaluate(model, corpus, tokenizer, config)
        checkpoint("checkpoint.pt", config["steps"])
        write_json(run.path/"metrics.json", {"initial": initial, "final": final, "updates": updates,
                   "parameters": sum(p.numel() for p in model.parameters()), "scored_tokens_total": scored_tokens,
                   "scored_tokens_this_run": sum(u["scored_tokens"] for u in updates), "update_seconds": update_seconds,
                   "scored_tokens_per_update_second": sum(u["scored_tokens"] for u in updates)/update_seconds,
                   "scope": "local CPU integration timing including data/batching; excludes checkpoint/evaluation; not VIT throughput"})
    return run.path


def load_pretrained(path):
    import torch
    from .neural import CausalLM, ModelConfig, read_checkpoint
    record = read_checkpoint(path)
    require(record.get("kind") == "pretraining_lm" and record.get("schema") == "aim-pretraining-v1", "not a pretraining LM")
    tokenizer = tokenizer_from_spec(record["tokenizer"])
    cfg = ModelConfig(**record["model_config"])
    require(cfg.vocab_size == tokenizer.vocab_size and cfg.parameter_estimate() <= 2_000_000 and cfg.context <= 512, "pretraining model/tokenizer allocation mismatch")
    model = CausalLM(cfg)
    model.load_state_dict(record["model"])
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "nonfinite pretrained weights")
    model.eval()
    return model, tokenizer, record

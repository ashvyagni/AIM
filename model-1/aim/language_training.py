"""Opt-in tokenizer-aware SFT/DPO/bounded RLVR; independent from the legacy runner."""
import copy
from dataclasses import asdict
from pathlib import Path

import torch

from .corpus import require
from .language_contract import ENCODING, KIND, SCHEMA, model_config, output_contract, read_source, tokenizer_for, validate_initialization
from .language_data import prepare, rewards
from .neural import CausalLM
from .objectives import dpo_loss, reinforce_loss
from .response_encoding import encoding_manifest, scores
from .tracking import Run, digest, file_hash, write_json
from .training import save_checkpoint, seed_all


def validate_config(config):
    from .training import validate_config as common
    required = {"stage", "task", "seed", "steps", "batch_size", "learning_rate", "weight_decay", "checkpoint_every", "eval_batch_size", "model"}
    require(isinstance(config, dict) and required <= set(config) <= required | {"dataset_path", "beta", "kl_coefficient"}, "language configuration fields differ")
    require(config["stage"] in {"sft", "preference", "rlvr"}, "Judge stays a separate trainer")
    require(config["task"] in {"arithmetic", "symbolic", "external", "structured"}, "unsupported language task")
    common({**config, "task": "arithmetic"})
    require(config["steps"] <= 1000 and config["batch_size"] <= 16, "language training exceeds miniature step/batch budget")
    for key, limit in (("checkpoint_every", 1000), ("eval_batch_size", 16)):
        require(type(config[key]) is int and 1 <= config[key] <= limit, "invalid "+key)
    require(config["seed"] < 2**32, "seed exceeds numpy reproducibility range")
    if "beta" in config:
        require(config["stage"] == "preference" and config["beta"] > 0, "positive beta belongs to preference training only")
    if "kl_coefficient" in config:
        require(config["stage"] == "rlvr", "KL coefficient belongs to RLVR only")


@torch.no_grad()
def evaluate(model, tokenizer, rows, batch_size):
    was_training = model.training
    model.eval()
    nll, count, byte_count, correct, preference_count = 0., 0, 0, 0, 0
    try:
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset+batch_size]
            logp, counts, encodings = scores(model, tokenizer, [(r["prompt"], r["response"]) for r in batch])
            nll -= float(logp.sum())
            count += int(counts.sum())
            byte_count += sum(r["response_bytes"] for r in encodings)
            pairs = [r for r in batch if "chosen" in r and "rejected" in r]
            if pairs:
                chosen, _, _ = scores(model, tokenizer, [(r["prompt"], r["chosen"]) for r in pairs])
                rejected, _, _ = scores(model, tokenizer, [(r["prompt"], r["rejected"]) for r in pairs])
                correct += int((chosen > rejected).sum())
                preference_count += len(pairs)
    finally:
        model.train(was_training)
    require(count > 0, "empty response metrics")
    return {"response_token_nll": nll/count, "scored_tokens": count, "response_bytes": byte_count,
            "preference_correct": correct, "preference_examples": preference_count,
            "preference_accuracy": correct/preference_count if preference_count else None,
            "examples": len(rows), "scope": "fixed validation responses; no generation/verifier capability or cross-tokenizer NLL comparison"}


def objective(model, reference, tokenizer, batch, config):
    stage = config["stage"]
    if stage == "sft":
        values, counts, encodings = scores(model, tokenizer, [(r["prompt"], r["response"]) for r in batch])
        return -values.sum()/counts.sum(), {"scored_tokens": int(counts.sum()), "response_bytes": sum(r["response_bytes"] for r in encodings)}
    if stage == "preference":
        chosen, rejected = ([(r["prompt"], r[key]) for r in batch] for key in ("chosen", "rejected"))
        pc, cc, _ = scores(model, tokenizer, chosen)
        pr, cr, _ = scores(model, tokenizer, rejected)
        with torch.no_grad():
            rc, _, _ = scores(reference, tokenizer, chosen)
            rr, _, _ = scores(reference, tokenizer, rejected)
        return dpo_loss(pc, pr, rc, rr, config.get("beta", .1)), {
            "scored_tokens": int(cc.sum()+cr.sum()), "policy_logprob_margin": float((pc-pr).detach().mean()),
            "label_origins": sorted({r["label_origin"] for r in batch})}
    losses, sampled, kls, actions, tokens, calls = [], [], [], [], 0, 0
    for row in batch:
        pairs = [(row["prompt"], answer) for answer in row["candidates"]]
        logits, counts, _ = scores(model, tokenizer, pairs)
        with torch.no_grad():
            ref, _, _ = scores(reference, tokenizer, pairs)
        verified = torch.tensor(rewards(config["task"], row), dtype=torch.float32)
        action = int(torch.multinomial(logits.detach().softmax(-1), 1))
        loss, kl = reinforce_loss(logits, ref, verified, action, config.get("kl_coefficient", .02))
        losses.append(loss); sampled.append(float(verified[action])); kls.append(float(kl.detach())); actions.append(action)
        tokens += int(counts.sum()); calls += len(pairs)
    return torch.stack(losses).mean(), {"scored_tokens": tokens, "sample_reward": sum(sampled)/len(sampled),
           "candidate_kl": sum(kls)/len(kls), "sampled_actions": actions, "verifier_calls": calls,
           "scope": "finite-candidate REINFORCE; no free-generation RL claim"}


def train(config, runs, tokenizer=None, initialize=None, resume=None):
    paths = [Path(p) for p in (initialize, resume, config.get("dataset_path")) if p]
    with Run(Path(runs), "language-"+str(config.get("stage", "invalid")),
             {"config": config, "explicit_tokenizer": tokenizer.specification() if tokenizer else None,
              "initialize": str(initialize) if initialize else None, "resume": str(resume) if resume else None}, paths) as run:
        validate_config(config)
        require(not (initialize and resume), "choose initialization or resume")
        seed_all(config["seed"])
        source = read_source(resume or initialize) if (resume or initialize) else None
        tokenizer = tokenizer_for(source, tokenizer)
        cfg = model_config(config["model"], tokenizer)
        if source:
            validate_initialization(source, cfg, tokenizer, config["task"], config["stage"], bool(resume))
        require(config["stage"] == "sft" or source is not None, "post-training stage needs a compatible AIM checkpoint")
        data = prepare(config["task"], config["stage"], config.get("dataset_path"))
        encoded = encoding_manifest({s: data[s] for s in ("train", "validation")}, tokenizer, cfg.context, config["stage"])
        write_json(run.path/"dataset.json", data)
        write_json(run.path/"tokenizer.json", tokenizer.specification())
        write_json(run.path/"encoding-manifest.json", encoded)
        model = CausalLM(cfg)
        if source:
            model.load_state_dict(source["model"])
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "nonfinite initialization")
        reference = copy.deepcopy(model).eval()
        for parameter in reference.parameters(): parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        stable = {k: v for k, v in config.items() if k != "steps"}
        start, total = 0, 0
        if resume:
            require(source["stable_config"] == stable and source["dataset_hash"] == data["dataset_hash"] and
                    source["encoding_manifest_hash"] == encoded["encoding_manifest_hash"], "resume changes configuration/data/encoding")
            start, total = source["step"], source["scored_tokens"]
            require(start < config["steps"], "resume must advance optimizer steps")
            optimizer.load_state_dict(source["optimizer"])
            reference.load_state_dict(source["reference"])
            torch.set_rng_state(source["torch_rng"])
        initial = evaluate(model, tokenizer, data["validation"], config["eval_batch_size"])
        write_json(run.path/"initial-metrics.json", initial)
        parent_hash = file_hash(resume or initialize) if source else None

        def checkpoint(filename, step):
            record = {"schema": SCHEMA, "kind": KIND, "origin": "aim-random-init-v1", "task": config["task"],
                      "stage": config["stage"], "model_config": asdict(cfg), "model": model.state_dict(), "reference": reference.state_dict(),
                      "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(), "step": step, "scored_tokens": total,
                      "stable_config": stable, "dataset_hash": data["dataset_hash"], "encoding": ENCODING,
                      "encoding_manifest_hash": encoded["encoding_manifest_hash"], "tokenizer": tokenizer.specification(),
                      "tokenizer_hash": digest(tokenizer.specification()), "output_contract": output_contract(config["task"]),
                      "research_contract": output_contract(config["task"]) if config["task"] in {"structured", "symbolic"} else None,
                      "parent_sha256": parent_hash}
            save_checkpoint(run, record, filename)

        updates = []
        model.train()
        for step in range(start, config["steps"]):
            batch = [data["train"][(step*config["batch_size"]+i)%len(data["train"])] for i in range(config["batch_size"])]
            optimizer.zero_grad()
            loss, details = objective(model, reference, tokenizer, batch, config)
            require(bool(torch.isfinite(loss)), "nonfinite language objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += details["scored_tokens"]
            row = {"step": step+1, "loss": float(loss.detach()), "gradient_norm": float(norm), "batch_hash": digest(batch), **details}
            updates.append(row)
            run.metric(step+1, **{k: v for k, v in row.items() if k != "step"})
            if (step+1) % config["checkpoint_every"] == 0:
                checkpoint(f"checkpoint-step-{step+1:06d}.pt", step+1)
        final = evaluate(model, tokenizer, data["validation"], config["eval_batch_size"])
        checkpoint("checkpoint.pt", config["steps"])
        write_json(run.path/"metrics.json", {"initial": initial, "final": final, "updates": updates,
                   "parameters": cfg.parameter_estimate(), "tokenizer_hash": digest(tokenizer.specification()),
                   "scored_tokens_total": total, "scored_tokens_this_run": sum(r["scored_tokens"] for r in updates),
                   "scope": "opt-in miniature language objectives; token counts include candidate scoring for RLVR/preference"})
    return run.path

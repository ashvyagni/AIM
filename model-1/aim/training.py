"""Runnable miniature SFT, DPO, verifier-grounded REINFORCE and Judge calibration.

All learned weights originate locally. Fixtures validate mechanisms, not research
capability. Checkpoints preserve optimizer/RNG/data state for exact CPU resume.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .contracts import ContractError
from .datasets import VERSION, arithmetic_data, arithmetic_reward, assert_disjoint, judge_data, load_supervised_dataset
from .judge import FEATURE_VERSION, TARGET
from .metrics import calibration_metrics
from .neural import ByteTokenizer, CausalLM, DecisionNetwork, ModelConfig, read_checkpoint
from .objectives import dpo_loss, proper_loss, reinforce_loss
from .tracking import Run, digest, file_hash, write_json


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def sequence_scores(model, tokenizer, pairs, *, pad_to=None):
    sequences, starts = [], []
    for prompt, response in pairs:
        prefix = [tokenizer.bos_id] + tokenizer.encode(prompt)
        seq = prefix + tokenizer.encode(response) + [tokenizer.eos_id]
        if len(seq) - 1 > model.cfg.context:
            raise ContractError("Training sequence exceeds context; truncation is forbidden")
        sequences.append(seq)
        starts.append(len(prefix)-1)
    width = max(map(len, sequences))
    if pad_to is not None:
        if type(pad_to) is not int or not width-1<=pad_to<=model.cfg.context:
            raise ContractError("Invalid fixed training input length")
        width=pad_to+1
    batch = torch.full((len(sequences),width), tokenizer.pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences),width-1), dtype=torch.bool)
    for i, seq in enumerate(sequences):
        batch[i,:len(seq)] = torch.tensor(seq)
        mask[i, starts[i]:len(seq)-1] = True
    logits = model(batch[:,:-1])
    logp = logits.log_softmax(-1).gather(-1, batch[:,1:,None]).squeeze(-1)
    return (logp * mask).sum(-1), mask.sum(-1)


def validate_config(config):
    if config.get("task", "arithmetic") not in {"arithmetic", "symbolic"}:
        raise ContractError("Unsupported training task")
    if config.get("task") == "symbolic" and (config.get("stage") == "judge" or config.get("dataset_path")):
        raise ContractError("Symbolic LM stages use versioned procedural data; Judge is a separate trainer")
    if config.get("stage") not in {"sft", "preference", "rlvr", "judge"}:
        raise ContractError("Stage must be sft, preference, rlvr or judge; combined objective unavailable")
    for k in ("steps", "batch_size", "seed"):
        if type(config.get(k)) is not int or config[k] < (0 if k=="seed" else 1):
            raise ContractError(f"Invalid {k}")
    if config["steps"] > 10000 or config["batch_size"] > 64:
        raise ContractError("Miniature harness budget exceeded")
    for k in ("learning_rate", "weight_decay", "beta", "kl_coefficient"):
        value = config.get(k, 0.0)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
            raise ContractError(f"Invalid {k}")
    if config.get("learning_rate",0) <= 0:
        raise ContractError("Learning rate must be positive")


def train(config, runs, *, initialize=None, resume=None):
    inputs = [Path(p) for p in (initialize,resume,config.get("dataset_path")) if p]
    with Run(Path(runs), "train-"+str(config.get("stage","invalid")),
             {**config, "initialize": str(initialize) if initialize else None, "resume": str(resume) if resume else None}, inputs) as run:
        validate_config(config)
        if initialize and resume:
            raise ContractError("Choose initialization or resume, not both")
        seed_all(config["seed"])
        if config["stage"] == "judge":
            if initialize:
                raise ContractError("Judge has separate random initialization; use resume for its own checkpoints")
            _train_judge(config,run,resume)
        else:
            _train_lm(config,run,initialize,resume)
    return run.path


def save_checkpoint(run, record, filename="checkpoint.pt"):
    path = run.path/filename
    with path.open("xb") as f:
        torch.save(record,f)
    write_json(run.path/(filename+".sha256.json"), {"sha256":file_hash(path)})


def _train_lm(config, run, initialize, resume):
    stage = config["stage"]
    cfg = ModelConfig(**config.get("model",{}))
    if cfg.parameter_estimate() > 2_000_000:
        raise ContractError("Model-1 smoke harness caps allocations at 2M parameters; scale trials need a separate reviewed run plan")
    tokenizer = ByteTokenizer()
    if cfg.vocab_size != tokenizer.vocab_size:
        raise ContractError("Model vocab must match the engineering tokenizer")
    symbolic = config.get("task") == "symbolic"
    if symbolic:
        from .symbolic_data import symbolic_data, VERSION as SYMBOLIC_VERSION
        data, validation, data_version = symbolic_data("train"), symbolic_data("validation"), SYMBOLIC_VERSION
    elif config.get("dataset_path"):
        if stage == "rlvr":
            raise ContractError("External RLVR datasets need a reviewed verifier adapter; arithmetic fixture only")
        data, validation, data_version = load_supervised_dataset(config["dataset_path"])
    else:
        data, validation, data_version = arithmetic_data("train"), arithmetic_data("validation",32), VERSION
    assert_disjoint(data,validation)
    data_hash = digest({"train":data,"validation":validation,"version":data_version})
    write_json(run.path/"dataset.json", {"version":data_version,"train":data,"validation":validation,"dataset_hash":data_hash})
    write_json(run.path/"tokenizer.json", tokenizer.specification())
    model = CausalLM(cfg)
    source = read_checkpoint(resume or initialize) if (resume or initialize) else None
    if source:
        pretraining_initialization = bool(initialize and stage == "sft" and source.get("kind") == "pretraining_lm")
        if (source["kind"] != "causal_lm" and not pretraining_initialization) or source["model_config"] != asdict(cfg):
            raise ContractError("Checkpoint/model architecture mismatch")
        if source.get("tokenizer") != tokenizer.specification():
            raise ContractError("SFT/post-training requires the exact byte-tokenizer contract; no implicit vocabulary conversion")
        if pretraining_initialization and source.get("schema") != "aim-pretraining-v1":
            raise ContractError("Unsupported pretraining checkpoint schema")
        from .symbolic_loop import RESEARCH_CONTRACT
        if not pretraining_initialization and symbolic != (source.get("research_contract") == RESEARCH_CONTRACT):
            raise ContractError("Symbolic and legacy training checkpoint contracts cannot be mixed")
        model.load_state_dict(source["model"])
    if stage in {"preference","rlvr"} and source is None:
        raise ContractError("Post-training requires an AIM-owned initial checkpoint")
    reference = copy.deepcopy(model).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config.get("weight_decay",0.01))
    stable_config = {k:v for k,v in config.items() if k != "steps"}
    start = 0
    if resume:
        if source["stable_config"] != stable_config or source["dataset_hash"] != data_hash:
            raise ContractError("Resume changes configuration or training data")
        start = source["step"]
        if start >= config["steps"]:
            raise ContractError("Resume target must exceed checkpoint step")
        optimizer.load_state_dict(source["optimizer"])
        reference.load_state_dict(source["reference"])
        torch.set_rng_state(source["torch_rng"])
    initial = lm_metrics(model,reference,tokenizer,validation)
    run.metric(start,validation=initial)
    write_json(run.path/"initial-metrics.json",initial)
    all_metrics = []
    for step in range(start, config["steps"]):
        batch = [data[(step*config["batch_size"]+i)%len(data)] for i in range(config["batch_size"])]
        optimizer.zero_grad()
        if stage == "sft":
            scores, counts = sequence_scores(model,tokenizer,[(row["prompt"],row["response"]) for row in batch])
            loss = -scores.sum()/counts.sum()
            extra = {"supervised_tokens":int(counts.sum())}
        elif stage == "preference":
            chosen = [(r["prompt"],r["chosen"]) for r in batch]
            rejected = [(r["prompt"],r["rejected"]) for r in batch]
            pc,_ = sequence_scores(model,tokenizer,chosen)
            pr,_ = sequence_scores(model,tokenizer,rejected)
            with torch.no_grad():
                rc,_ = sequence_scores(reference,tokenizer,chosen)
                rr,_ = sequence_scores(reference,tokenizer,rejected)
            loss = dpo_loss(pc,pr,rc,rr,config.get("beta",0.1))
            extra = {"preference_margin":float((pc-pr).detach().mean()),
                     "label_origins":sorted({r["label_origin"] for r in batch})}
        else:
            losses,rewards,kls,entropies = [],[],[],[]
            for row in batch:
                if symbolic:
                    from .symbolic_data import symbolic_reward
                    candidates = row["candidates"]
                    reward_function = symbolic_reward
                else:
                    target = sum(row["operands"])
                    candidates = [str(target-1),str(target),str(target+1)]
                    reward_function = arithmetic_reward
                pairs = [(row["prompt"],x) for x in candidates]
                logits,_ = sequence_scores(model,tokenizer,pairs)
                with torch.no_grad():
                    reference_logits,_ = sequence_scores(reference,tokenizer,pairs)
                # Candidate set is a bounded environment; no free-form RL claim.
                reward = torch.tensor([reward_function(row,x) for x in candidates])
                action = torch.multinomial(logits.detach().softmax(-1),1).item()
                loss_i,kl = reinforce_loss(logits,reference_logits,reward,action,config.get("kl_coefficient",0.02))
                losses.append(loss_i)
                rewards.append(reward[action].item())
                kls.append(kl.detach().item())
                entropies.append(float(-(logits.softmax(-1)*logits.log_softmax(-1)).sum().detach()))
            loss = torch.stack(losses).mean()
            extra = {"sample_reward":sum(rewards)/len(rewards),"candidate_kl":sum(kls)/len(kls),
                     "entropy":sum(entropies)/len(entropies),"verifier_calls":3*len(batch)}
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
        metrics = {"loss":loss.item(),"grad_norm":float(norm), **extra}
        run.metric(step+1,**metrics)
        all_metrics.append({"step":step+1,**metrics})
    final = lm_metrics(model,reference,tokenizer,validation)
    write_json(run.path/"metrics.json", {"initial":initial,"final":final,"updates":all_metrics,
               "parameters":sum(p.numel() for p in model.parameters()),
               "scope":"miniature training mechanism; no capability claim", "dataset_version":data_version})
    record = {"kind":"causal_lm","origin":"aim-random-init-v1","model_config":asdict(cfg),
              "model":model.state_dict(),"optimizer":optimizer.state_dict(),"reference":reference.state_dict(),
              "torch_rng":torch.get_rng_state(),"step":config["steps"],"stage":stage,"stable_config":stable_config,
              "dataset_hash":data_hash,"tokenizer":tokenizer.specification(),"parent_sha256":file_hash(resume or initialize) if source else None}
    if symbolic:
        from .symbolic_loop import RESEARCH_CONTRACT
        record["research_contract"] = RESEARCH_CONTRACT
    save_checkpoint(run,record)


@torch.no_grad()
def lm_metrics(model, reference, tokenizer, rows):
    scores,counts = sequence_scores(model,tokenizer,[(r["prompt"],r["response"]) for r in rows])
    rejected,_ = sequence_scores(model,tokenizer,[(r["prompt"],r["rejected"]) for r in rows])
    expected=[]
    for row in rows:
        if "candidates" in row and "lhs" in row:
            from .symbolic_data import symbolic_reward
            options = row["candidates"]
            logp, _ = sequence_scores(model, tokenizer, [(row["prompt"], x) for x in options])
            reward = torch.tensor([symbolic_reward(row, x) for x in options])
            expected.append(float((logp.softmax(-1)*reward).sum()))
            continue
        if "operands" not in row:
            continue
        n = sum(row["operands"])
        options=[str(n-1),str(n),str(n+1)]
        logp,_=sequence_scores(model,tokenizer,[(row["prompt"],x) for x in options])
        expected.append(logp.softmax(-1)[1].item())
    return {"response_token_nll":float(-scores.sum()/counts.sum()),
            "preference_accuracy":float((sequence_scores(model,tokenizer,[(r["prompt"],r["chosen"]) for r in rows])[0]>rejected).float().mean()),
            "bounded_candidate_expected_reward":sum(expected)/len(expected) if expected else None,"n":len(rows),
            "label_origins":sorted({r["label_origin"] for r in rows})}


def _train_judge(config, run, resume):
    datasets = {s:judge_data(s,n) for s,n in [("train",256),("validation",96),("test",96)]}
    datasets["ood"] = judge_data("test",96,ood=True)
    assert_disjoint(*datasets.values())
    data_hash = digest(datasets)
    write_json(run.path/"dataset.json", {"version":VERSION,"datasets":datasets,"hash":data_hash,
               "label_definition":TARGET,"feature_version":FEATURE_VERSION})
    model = DecisionNetwork()
    optimizer = torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config.get("weight_decay",0.01))
    stable_config = {k:v for k,v in config.items() if k != "steps"}
    start = 0
    if resume:
        record=read_checkpoint(resume)
        if record["kind"] != "decision_network" or record["stable_config"] != stable_config or record["dataset_hash"] != data_hash:
            raise ContractError("Judge resume configuration/dataset mismatch")
        model.load_state_dict(record["model"])
        optimizer.load_state_dict(record["optimizer"])
        torch.set_rng_state(record["torch_rng"])
        start=record["step"]
        if start >= config["steps"]:
            raise ContractError("Resume target must advance Judge steps")
    tensors = {s:(torch.tensor([r["features"] for r in rows],dtype=torch.float32),
                  torch.tensor([r["label"] for r in rows],dtype=torch.float32)) for s,rows in datasets.items()}
    with torch.no_grad():
        initial={s:calibration_metrics(model(x).sigmoid().tolist(), y.tolist()) for s,(x,y) in tensors.items() if s!="train"}
    write_json(run.path/"initial-metrics.json",initial)
    x,y=tensors["train"]
    for step in range(start,config["steps"]):
        indices=torch.tensor([(step*config["batch_size"]+i)%len(x) for i in range(config["batch_size"])])
        optimizer.zero_grad()
        loss=proper_loss(model(x[indices]),y[indices],config.get("scoring_rule","log"))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite judge loss")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
        run.metric(step+1,loss=loss.item(),grad_norm=float(norm))
    with torch.no_grad():
        vx,vy=tensors["validation"]
        temperatures=[0.5,0.75,1.0,1.25,1.5,2.0,3.0,5.0]
        temperature=min(temperatures,key=lambda t:proper_loss(model(vx)/t,vy,"log").item())
        final={s:{"uncalibrated":calibration_metrics(model(x).sigmoid().tolist(),y.tolist()),
                  "temperature_scaled":calibration_metrics((model(x)/temperature).sigmoid().tolist(),y.tolist())}
               for s,(x,y) in tensors.items() if s!="train"}
    metrics={"initial":initial,"final":final,"temperature":temperature,"parameters":sum(p.numel() for p in model.parameters()),
             "scope":"synthetic next-measurement forecasts; OOD cubic diagnostic only"}
    write_json(run.path/"metrics.json",metrics)
    save_checkpoint(run,{"origin":"aim-random-init-v1","kind":"decision_network","model":model.state_dict(),
                         "optimizer":optimizer.state_dict(),"torch_rng":torch.get_rng_state(),"step":config["steps"],
                         "hidden":32,"temperature":temperature,"feature_version":FEATURE_VERSION,"target":TARGET,
                         "dataset_hash":data_hash,"stable_config":stable_config,"metrics":metrics})

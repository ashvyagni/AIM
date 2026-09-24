"""Phase 2A SFT: random initialization, validation-only selection, frozen choices."""
import argparse
import json
import time
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import torch

from .contracts import ContractError
from .neural import ByteTokenizer, CausalLM, ModelConfig, read_checkpoint
from .research_data import load_training_data
from .research_format import VERSION, ResearchOutputError, parse_hypothesis
from .training import save_checkpoint, seed_all, sequence_scores
from .tracking import ROOT, Run, digest, file_hash, write_json


def score_generation(text, row):
    try:
        hypothesis=parse_hypothesis(text,row["aliases"])[0]
    except ResearchOutputError as exc:
        return {"valid":False,"agrees":False,"error_category":exc.category,"absolute_error":None}
    x=Fraction(row["case"]["target_x"])
    prediction=sum(Fraction(str(c))*x**i for i,c in enumerate(hypothesis.coefficients))
    measurement=Fraction(str(row["case"]["experiment"]["observations"][0][1]))
    error=abs(prediction-measurement)
    return {"valid":True,"agrees":error<=Fraction(1,100_000_000),"error_category":None,"absolute_error":float(error)}


@torch.no_grad()
def validation_metrics(model,tokenizer,rows,max_new_tokens):
    model.eval()
    total_score,total_tokens=0.0,0
    for start in range(0,len(rows),16):
        batch=rows[start:start+16]
        score,count=sequence_scores(model,tokenizer,[(r["prompt"],r["response"]) for r in batch])
        total_score+=float(score.sum());total_tokens+=int(count.sum())
    outputs=[]
    for row in rows:
        text=model.generate_text(tokenizer,row["prompt"],max_new_tokens)
        outputs.append({"id":row["id"],"raw_output":text,**score_generation(text,row)})
    model.train()
    return {"n":len(rows),"response_token_nll":-total_score/total_tokens,
            "valid_rate":sum(r["valid"] for r in outputs)/len(rows),
            "prediction_agreement_rate":sum(r["agrees"] for r in outputs)/len(rows),"outputs":outputs}


def selection_key(metrics,step):
    return (metrics["prediction_agreement_rate"],metrics["valid_rate"],-metrics["response_token_nll"],-step)


def validate_config(config):
    if config.get("protocol")!=VERSION:
        raise ContractError("Unknown structured research protocol")
    steps=config.get("steps")
    if type(steps) is not int or not 1<=steps<=1800:
        raise ContractError("Research SFT allows 1..1800 bounded updates")
    seeds=config.get("seeds")
    if not isinstance(seeds,list) or not 1<=len(seeds)<=5 or len(set(seeds))!=len(seeds) or any(type(s) is not int or s<0 for s in seeds):
        raise ContractError("Invalid independent seeds")
    points=config.get("evaluation_steps")
    if not isinstance(points,list) or not points or any(type(x) is not int or not 1<=x<=steps for x in points) or sorted(set(points))!=points or points[-1]!=steps:
        raise ContractError("Evaluation steps must be ordered, unique and end at total steps")
    for key,limit in (("batch_size",32),("max_new_tokens",128),("deadline_seconds",1200)):
        if type(config.get(key)) is not int or not 1<=config[key]<=limit:
            raise ContractError(f"Invalid bounded {key}")
    from .contracts import finite_number
    if not 0<finite_number(config["learning_rate"])<=0.1 or not 0<=finite_number(config["weight_decay"])<=1:
        raise ContractError("Invalid optimizer configuration")


def restore_history(source, resume_path):
    """Validate prior selection evidence, including original v1 checkpoint layouts."""
    history=list(source.get("evaluation_history",[]))
    expected=([0]+[s for s in source["configuration"]["evaluation_steps"] if s<source["step"]]) if source["step"] else []
    if "evaluation_history" not in source:
        # Legacy checkpoints predate embedded history. Reconstruct only from
        # their original sibling artifacts; missing evidence is an error.
        for step in expected:
            checkpoint=Path(resume_path).parent/f"checkpoint-step{step:04d}.pt"
            validation=checkpoint.parent/f"validation-step{step:04d}.json"
            metrics=json.loads(validation.read_text())
            read_checkpoint(checkpoint)  # also checks the original sidecar
            history.append({"step":step,"checkpoint":str(checkpoint.resolve()),"sha256":file_hash(checkpoint),
                "validation_path":str(validation.resolve()),"validation_sha256":file_hash(validation),
                **{k:v for k,v in metrics.items() if k!="outputs"}})
    if history and history[-1]["step"]==source["step"]:
        expected.append(source["step"])
    if [r["step"] for r in history]!=expected:
        raise ContractError("Incomplete or reordered checkpoint-selection history")
    for row in history:
        if file_hash(row["checkpoint"])!=row["sha256"] or file_hash(row["validation_path"])!=row["validation_sha256"]:
            raise ContractError("Historical selection artifact changed")
        metrics=json.loads(Path(row["validation_path"]).read_text())
        if any(row[k]!=v for k,v in metrics.items() if k!="outputs"):
            raise ContractError("Historical validation summary changed")
    return history


def train_seed(config,data,data_path,runs,seed,protocol_path,*,resume=None):
    inputs=[data_path,protocol_path]+([Path(resume)] if resume else [])
    with Run(Path(runs),f"research-sft-seed{seed}",{**config,"seed":seed,"resume":str(resume) if resume else None},inputs) as run:
        validate_config(config)
        seed_all(seed)
        cfg=ModelConfig(**config["model"])
        tokenizer=ByteTokenizer()
        if cfg.parameter_estimate()>2_000_000 or cfg.vocab_size!=tokenizer.vocab_size:
            raise ContractError("Model exceeds reference allocation/tokenizer contract")
        train_rows,validation=data["train"],data["validation"]
        for row in train_rows+validation:
            if 1+len(tokenizer.encode(row["prompt"]))+config["max_new_tokens"]>cfg.context:
                raise ContractError("Generation budget exceeds declared context")
        model=CausalLM(cfg)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
        sampler=torch.Generator().manual_seed(seed)
        dataset_hash=digest(data)
        write_json(run.path/"tokenizer.json",tokenizer.specification())
        evaluations=[];best=None;parent_hash=None;step=0
        if resume:
            source=read_checkpoint(resume)
            if (source.get("stage")!="research-sft" or source.get("research_contract")!=VERSION or
                    source.get("seed")!=seed or source.get("dataset_hash")!=dataset_hash or
                    source.get("model_config")!=asdict(cfg) or source.get("tokenizer")!=tokenizer.specification()):
                raise ContractError("Resume stage, seed, data, model or tokenizer mismatch")
            previous=source["configuration"]
            mutable={"steps","evaluation_steps","deadline_seconds","seeds"}
            if {k:v for k,v in previous.items() if k not in mutable}!={k:v for k,v in config.items() if k not in mutable}:
                raise ContractError("Resume cannot change optimization or generation configuration")
            step=source["step"]
            if type(step) is not int or not 0<=step<config["steps"]:
                raise ContractError("Resume must advance beyond the saved update")
            if ([s for s in previous["evaluation_steps"] if s<=step]!=
                    [s for s in config["evaluation_steps"] if s<=step]):
                raise ContractError("Resume cannot rewrite past evaluation points")
            evaluations=restore_history(source,resume)
            best=max(evaluations,key=lambda r:selection_key(r,r["step"]),default=None)
            model.load_state_dict(source["model"])
            optimizer.load_state_dict(source["optimizer"])
            sampler.set_state(source["sampler_rng"])
            torch.set_rng_state(source["torch_rng"])
            parent_hash=file_hash(resume)
            run.event("CHECKPOINT_RESUMED",{"checkpoint":str(Path(resume).resolve()),"sha256":parent_hash,
                "step":step,"historical_evaluations":len(evaluations)})

        def checkpoint(filename):
            nonlocal parent_hash
            save_checkpoint(run,{"kind":"causal_lm","origin":"aim-random-init-v1","research_contract":VERSION,
                "model_config":asdict(cfg),"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                "torch_rng":torch.get_rng_state(),"sampler_rng":sampler.get_state(),"step":step,"stage":"research-sft",
                "configuration":config,"seed":seed,"dataset_hash":dataset_hash,"tokenizer":tokenizer.specification(),
                "parent_sha256":parent_hash,"evaluation_history":list(evaluations)},filename)
            path=run.path/filename
            parent_hash=file_hash(path)
            return path

        def evaluate():
            nonlocal best
            path=checkpoint(f"checkpoint-step{step:04d}.pt")
            metrics=validation_metrics(model,tokenizer,validation,config["max_new_tokens"])
            validation_path=run.path/f"validation-step{step:04d}.json"
            write_json(validation_path,metrics)
            summary={k:v for k,v in metrics.items() if k!="outputs"}
            record={"step":step,"checkpoint":str(path.resolve()),"sha256":file_hash(path),
                    "validation_path":str(validation_path.resolve()),"validation_sha256":file_hash(validation_path),**summary}
            evaluations.append(record)
            run.metric(step,validation=summary)
            if best is None or selection_key(metrics,step)>selection_key(best,best["step"]):
                best=record
            print(json.dumps({"seed":seed,"step":step,"validation":summary,"run":str(run.path)}),flush=True)

        started=time.monotonic()
        # Saved checkpoints precede their validation. Recompute that boundary
        # deterministically; never replace the original random baseline.
        if (not resume or step==0 or step in config["evaluation_steps"]) and (not evaluations or evaluations[-1]["step"]!=step): evaluate()
        start_step=step
        for step in range(start_step+1,config["steps"]+1):
            if time.monotonic()-started>config["deadline_seconds"]:
                step-=1  # The next update has not happened yet.
                checkpoint("emergency-deadline.pt")
                raise TimeoutError("Research training deadline exceeded; emergency checkpoint retained")
            indices=torch.randint(len(train_rows),(config["batch_size"],),generator=sampler).tolist()
            batch=[train_rows[i] for i in indices]
            optimizer.zero_grad()
            scores,counts=sequence_scores(model,tokenizer,[(r["prompt"],r["response"]) for r in batch])
            loss=-scores.sum()/counts.sum()
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite research SFT loss")
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step()
            run.metric(step,loss=loss.item(),grad_norm=float(norm),supervised_tokens=int(counts.sum()))
            if step in config["evaluation_steps"]: evaluate()
        result={"seed":seed,"initial":evaluations[0],"selected":best,"evaluations":evaluations,
                "parameters":sum(p.numel() for p in model.parameters()),"dataset_hash":dataset_hash,
                "scope":"validation-only checkpoint selection; no test/OOD labels read"}
        write_json(run.path/"metrics.json",result)
    return result


def train_experiment(config,data_path,runs,*,resume=None):
    data_path=Path(data_path)
    protocol_path=ROOT/"docs/experiments/phase-2a-protocol.md"
    with Run(Path(runs),"research-training",{**config,"resume":str(resume) if resume else None},
             [data_path,protocol_path]+([Path(resume)] if resume else [])) as run:
        validate_config(config)
        if resume and len(config["seeds"])!=1:
            raise ContractError("Resume takes one checkpoint and exactly one matching seed")
        data=load_training_data(data_path)
        dataset_manifest=data_path.parent/"split-manifest.json"
        dataset_metadata=json.loads(dataset_manifest.read_text())
        if file_hash(data_path)!=dataset_metadata["train_validation_sha256"]:
            raise ContractError("Training data no longer matches split manifest")
        results=[train_seed(config,data,data_path,run.path/"seeds",s,protocol_path,resume=resume) for s in config["seeds"]]
        # Frozen before the separate evaluator is invoked; no test-driven reselection.
        if file_hash(data_path)!=dataset_metadata["train_validation_sha256"]:
            raise ContractError("Training data no longer matches split manifest")
        write_json(run.path/"selected-models.json",{"version":VERSION,"configuration":config,
            "protocol_sha256":file_hash(protocol_path),"data_path":str(data_path.resolve()),
            "data_sha256":file_hash(data_path),"holdout_sha256":dataset_metadata["holdout_sha256"],
            "selections":results,"selection_rule":"validation agreement, validity, NLL, earliest step"})
    return run.path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=Path("configs/structured-researcher.json"))
    parser.add_argument("--runs",type=Path,default=Path("runs"))
    parser.add_argument("--resume",type=Path,help="Continue one seed, retaining original selection history")
    args=parser.parse_args()
    print(train_experiment(json.loads(args.config.read_text()),args.data,args.runs,resume=args.resume))


if __name__=="__main__": main()

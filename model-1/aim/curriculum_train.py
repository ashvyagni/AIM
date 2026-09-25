"""Separate staged SFT recipes with explicit sample accounting and frozen selection."""
import argparse
import json
from pathlib import Path

from .contracts import ContractError
from .curriculum_tasks import EXPERIMENT, auxiliary_rows
from .finite_difference import CONTRACT, worked_response
from .research_train import train_seed, validate_config
from .tracking import ROOT, Run, digest, file_hash, write_json


def load_data(path):
    path=Path(path);data=json.loads(path.read_text());metadata=json.loads((path.parent/"split-manifest.json").read_text())
    allowed={"version","excluded_sha256","excluded_count","counts","group_hashes","train_validation_sha256",
             "holdout_sha256","ood_prompt_overlap_with_train","cross_version_world_overlap"}
    if not set(metadata)<=allowed or metadata.get("version")!=EXPERIMENT:
        raise ContractError("Unrecognized trainer metadata; holdout/audit labels are prohibited")
    if (set(data)!={"schema","version","train","validation"} or data["version"]!=EXPERIMENT or
            data["schema"]!="aim-curriculum-sft-v1" or file_hash(path)!=metadata["train_validation_sha256"]):
        raise ContractError("Curriculum data schema/hash mismatch")
    seen=set()
    for split in ("train","validation"):
        if not data[split]: raise ContractError("Empty research split")
        for row in data[split]:
            if row["split"]!=split or row["group"] in seen: raise ContractError("Wrong split or repeated world")
            seen.add(row["group"])
            if row["research_contract"]!=CONTRACT or row["response"]!=worked_response(row["world_coefficients"]):
                raise ContractError("Incorrect research teacher contract")
            if row["auxiliary"]!=auxiliary_rows(row): raise ContractError("Auxiliary teacher, world or split changed")
    return data,metadata


def train(config,data_path,runs):
    data_path=Path(data_path);protocol=ROOT/"docs/experiments/phase-2a2-protocol.md"
    with Run(Path(runs),"curriculum-training",config,[data_path,protocol]) as run:
        if config.get("protocol")!=EXPERIMENT: raise ContractError("Unknown curriculum protocol")
        for arm in ("worked","curriculum"):
            validate_config({**config,"arm":arm,"research_contract":CONTRACT})
        data,metadata=load_data(data_path);arms={}
        for arm in ("worked","curriculum"):
            arm_config={**config,"arm":arm,"research_contract":CONTRACT}
            selections=[train_seed(arm_config,data,data_path,run.path/arm,seed,protocol) for seed in config["seeds"]]
            arms[arm]={"configuration":arm_config,"selections":selections}
        if file_hash(data_path)!=metadata["train_validation_sha256"]: raise ContractError("Training data changed")
        from .neural import read_checkpoint
        import torch
        for left,right in zip(arms["worked"]["selections"],arms["curriculum"]["selections"]):
            a=read_checkpoint(left["initial"]["checkpoint"])["model"]
            b=read_checkpoint(right["initial"]["checkpoint"])["model"]
            if set(a)!=set(b) or any(not torch.equal(a[k],b[k]) for k in a): raise ContractError("Paired initial weights differ")
            a=read_checkpoint(left["evaluations"][-1]["checkpoint"])
            b=read_checkpoint(right["evaluations"][-1]["checkpoint"])
            if (left["sampling_digest"]!=right["sampling_digest"] or not torch.equal(a["sampler_rng"],b["sampler_rng"]) or
                left["processed_positions"]!=right["processed_positions"]): raise ContractError("Paired budgets or world draws differ")
        write_json(run.path/"selected-models.json",{"version":EXPERIMENT,"configuration":config,"arms":arms,
            "protocol_sha256":file_hash(protocol),"data_sha256":file_hash(data_path),"data_hash":digest(data),
            "holdout_sha256":metadata["holdout_sha256"],"paired_initial_weights_equal":True,
            "paired_world_draws_equal":True,"matched_dense_input_positions":True})
    return run.path


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=Path("configs/arithmetic-curriculum.json"))
    parser.add_argument("--runs",type=Path,default=Path("runs"));args=parser.parse_args()
    print(train(json.loads(args.config.read_text()),args.data,args.runs))

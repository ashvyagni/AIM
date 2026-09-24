"""Train both registered arms and freeze all selections without holdout access."""
import argparse
import copy
import json
from pathlib import Path

from .contracts import ContractError
from .finite_difference import EXPERIMENT, CONTRACT, worked_response
from .research_format import VERSION
from .research_train import train_seed
from .tracking import ROOT, Run, digest, file_hash, write_json


def arm_data(data,arm):
    if arm not in ("plain","worked"): raise ContractError("Unknown comparison arm")
    contract=VERSION if arm=="plain" else CONTRACT
    result=copy.deepcopy(data)
    for split in ("train","validation"):
        for row in result[split]:
            row["research_contract"]=contract
            if arm=="worked": row["response"]=worked_response(row["world_coefficients"])
    return result,contract


def train(config,data_path,runs):
    data_path=Path(data_path);protocol=ROOT/"docs/experiments/phase-2a1-protocol.md"
    with Run(Path(runs),"comparison-training",config,[data_path,protocol]) as run:
        if config.get("protocol")!=EXPERIMENT: raise ContractError("Wrong comparison protocol")
        data=json.loads(data_path.read_text());manifest=json.loads((data_path.parent/"split-manifest.json").read_text())
        if (set(data)!={"schema","version","train","validation"} or data["version"]!=EXPERIMENT or
            data["schema"]!="aim-research-comparison-sft-v1" or file_hash(data_path)!=manifest["train_validation_sha256"]):
            raise ContractError("Comparison dataset schema/hash mismatch")
        for split in ("train","validation"):
            if not data[split] or any(r["split"]!=split for r in data[split]): raise ContractError("Invalid training split")
        if {tuple(r["world_coefficients"]) for r in data["train"]}&{tuple(r["world_coefficients"]) for r in data["validation"]}:
            raise ContractError("Training/validation overlap")
        arms={}
        for arm in ("plain","worked"):
            rows,contract=arm_data(data,arm)
            arm_config={**config,"research_contract":contract}
            transformed=run.path/f"{arm}-train-validation.json";write_json(transformed,rows)
            selections=[train_seed(arm_config,rows,transformed,run.path/arm,s,protocol) for s in config["seeds"]]
            arms[arm]={"configuration":arm_config,"selections":selections,"data_sha256":file_hash(transformed)}
        if file_hash(data_path)!=manifest["train_validation_sha256"]: raise ContractError("Data changed during training")
        # Check shared tensors, not serialization hashes (checkpoint metadata differ).
        from .neural import read_checkpoint
        import torch
        for a,b in zip(arms["plain"]["selections"],arms["worked"]["selections"]):
            left=read_checkpoint(a["initial"]["checkpoint"])["model"]
            right=read_checkpoint(b["initial"]["checkpoint"])["model"]
            if any(not torch.equal(left[k],right[k]) for k in left): raise ContractError("Paired initial weights differ")
            if a["processed_positions"]!=b["processed_positions"]: raise ContractError("Paired compute-shape budgets differ")
        write_json(run.path/"selected-models.json",{"version":EXPERIMENT,"configuration":config,"arms":arms,
            "protocol_sha256":file_hash(protocol),"data_sha256":file_hash(data_path),
            "holdout_sha256":manifest["holdout_sha256"],"paired_initial_weights_equal":True,
            "matched_dense_input_positions":True,"data_hash":digest(data)})
    return run.path


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=Path("configs/finite-difference-comparison.json"))
    parser.add_argument("--runs",type=Path,default=Path("runs"));args=parser.parse_args()
    print(train(json.loads(args.config.read_text()),args.data,args.runs))

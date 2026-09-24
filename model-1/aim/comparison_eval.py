"""Frozen plain/worked comparison, with process diagnostics outside claim gates."""
import argparse
import json
import statistics
from pathlib import Path

from .contracts import ContractError
from .finite_difference import EXPERIMENT, CONTRACT, diagnostics
from .research_eval import SingleReferenceResearcher, run_backend, summarize, paired_comparison
from .researcher import TransformerResearcher
from .tracking import Run, file_hash, write_json


def process_summary(records,rows,contract):
    worlds={r["id"]:r for r in rows}
    for record in records:
        row=worlds[record["id"]]
        record["process"]=diagnostics(record["raw_output"],row["prompt"],row["aliases"],contract)
        record["output_utf8_bytes"]=len((record["raw_output"] or '').encode())
    n=len(records)
    return {"n":n,"coefficient_observation_accuracy":[sum(r["process"]["coefficient_observation_agreement"][i] for r in records)/n for i in range(3)],
        "d1_accuracy":sum(bool(r["process"]["d1_correct"]) for r in records)/n if contract==CONTRACT else None,
        "d2_accuracy":sum(bool(r["process"]["d2_correct"]) for r in records)/n if contract==CONTRACT else None,
        "all_steps_accuracy":sum(bool(r["process"]["all_steps_correct"]) for r in records)/n if contract==CONTRACT else None,
        "verified_with_incorrect_process":sum(r["verified"] and not r["process"]["all_steps_correct"] for r in records) if contract==CONTRACT else None,
        "mean_output_utf8_bytes":statistics.mean(r["output_utf8_bytes"] for r in records),
        "scope":"agreement with quadratic interpolation of observations; not hidden cubic coefficient recovery"}


def evaluate(selection_path,holdout_path,runs):
    selection_path=Path(selection_path);holdout_path=Path(holdout_path)
    with Run(Path(runs),"comparison-holdout",{"selection_sha256":file_hash(selection_path)},[selection_path,holdout_path]) as run:
        frozen=json.loads(selection_path.read_text())
        if frozen.get("version")!=EXPERIMENT or file_hash(holdout_path)!=frozen["holdout_sha256"]:
            raise ContractError("Comparison selection/holdout mismatch")
        if set(frozen["arms"])!={"plain","worked"}: raise ContractError("Both arms required")
        expected_seeds=frozen["configuration"]["seeds"]
        if not expected_seeds or len(set(expected_seeds))!=len(expected_seeds): raise ContractError("Invalid paired seeds")
        from .neural import read_checkpoint
        for arm in frozen["arms"].values():
            if [r["seed"] for r in arm["selections"]]!=expected_seeds: raise ContractError("Paired seeds differ")
            for seed in arm["selections"]:
                for phase in ("initial","selected"):
                    if file_hash(seed[phase]["checkpoint"])!=seed[phase]["sha256"]: raise ContractError("Frozen weights changed")
                    record=read_checkpoint(seed[phase]["checkpoint"])
                    if record["seed"]!=seed["seed"] or record["research_contract"]!=arm["configuration"]["research_contract"]:
                        raise ContractError("Frozen checkpoint identity mismatch")
        write_json(run.path/"selection-used.json",frozen)
        run.event("FROZEN_SELECTION_ACCEPTED",{"sha256":file_hash(selection_path)})
        holdout=json.loads(holdout_path.read_text())
        if holdout.get("version")!=EXPERIMENT or holdout.get("schema")!="aim-research-comparison-holdout-v1":
            raise ContractError("Invalid comparison holdout schema")
        seen=set()
        for split in ("test","ood"):
            if not holdout[split]: raise ContractError("Empty holdout split")
            for row in holdout[split]:
                if row["split"]!=split or row["group"] in seen: raise ContractError("Invalid or repeated holdout world")
                seen.add(row["group"])
        import torch
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        raw={};summary={};process={};gates={};initial_pairs={};arm_pairs={}
        inputs=[selection_path,holdout_path]
        for split in ("test","ood"):
            label="reference-"+split
            raw[label]=run_backend(holdout[split],SingleReferenceResearcher(),run,"reference",split,inputs)
            summary[label]=summarize(raw[label])
        for arm,details in frozen["arms"].items():
            gates[arm]={}
            for seed in details["selections"]:
                key=f"{arm}-seed{seed['seed']}"
                for phase in ("initial","selected"):
                    checkpoint=seed[phase]["checkpoint"]
                    researcher=TransformerResearcher(checkpoint,max_new_tokens=frozen["configuration"]["max_new_tokens"])
                    for split in ("test","ood"):
                        label=f"{key}-{phase}-{split}"
                        raw[label]=run_backend(holdout[split],researcher,run,f"{key}-{phase}",split,inputs+[Path(checkpoint)])
                        summary[label]=summarize(raw[label])
                        process[label]=process_summary(raw[label],holdout[split],details["configuration"]["research_contract"])
                        print(json.dumps({"backend":label,"verified":summary[label]["verified_worlds"],
                                          "valid_rate":summary[label]["valid_rate"]}),flush=True)
                pair=paired_comparison(raw[f"{key}-initial-test"],raw[f"{key}-selected-test"])
                initial_pairs[key]=pair
                score=summary[f"{key}-selected-test"]
                gates[arm][str(seed["seed"]) ]={"validity":score["valid_rate"]>=0.9,
                    "verified_success":score["verified_rate"]>=0.25,"initial_improvement":pair["verified_rate_change"]>=0.1}
        seeds=frozen["configuration"]["seeds"]
        for seed in seeds:
            arm_pairs[str(seed)]=paired_comparison(raw[f"plain-seed{seed}-selected-test"],raw[f"worked-seed{seed}-selected-test"])
        changes=[p["verified_rate_change"] for p in arm_pairs.values()]
        unbacked=sum(s["unbacked_verified_claims"] for s in summary.values())
        rates={arm:[summary[f"{arm}-seed{s}-selected-test"]["verified_rate"] for s in seeds] for arm in frozen["arms"]}
        arm_gates={arm:unbacked==0 and all(all(g.values()) for g in rows.values()) for arm,rows in gates.items()}
        report={"version":EXPERIMENT,"selection_sha256":file_hash(selection_path),"holdout_sha256":file_hash(holdout_path),
            "backends":summary,"process":process,"gates":gates,"arm_gate_passed":arm_gates,
            "initial_pairs":initial_pairs,"worked_vs_plain_pairs":arm_pairs,"unbacked_verified_claims":unbacked,
            "mean_worked_gain":statistics.mean(changes),"worked_advantage_gate":statistics.mean(changes)>=0.05 and min(changes)>=0,
            "worked_promotion_gate":arm_gates["worked"] and statistics.mean(changes)>=0.05 and min(changes)>=0,
            "seed_dispersion":{a:{"mean":statistics.mean(v),"min":min(v),"max":max(v),
                                    "sample_std":statistics.stdev(v) if len(v)>1 else None} for a,v in rates.items()},
            "scope":"shared finite synthetic worlds; diagnostic process checks never override final Controller status"}
        write_json(run.path/"case-results.json",raw);write_json(run.path/"metrics.json",report)
    return run.path,report


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--selection",type=Path,required=True)
    parser.add_argument("--holdout",type=Path,required=True);parser.add_argument("--runs",type=Path,default=Path("runs"))
    args=parser.parse_args();path,report=evaluate(args.selection,args.holdout,args.runs)
    print(json.dumps({"artifacts":str(path),"arm_gate_passed":report["arm_gate_passed"]}))

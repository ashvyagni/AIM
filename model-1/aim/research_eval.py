"""Frozen-checkpoint evaluation through actual Controller, tools and verifiers."""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

from .contracts import ContractError, Outcome, Status
from .controller import Controller
from .researcher import PolynomialResearcher, TransformerResearcher
from .research_format import VERSION
from .tracking import Run, file_hash, write_json


class SingleReferenceResearcher(PolynomialResearcher):
    model_id="single-quadratic-reference-v1"

    def hypothesize(self,state):
        return super().hypothesize(state)[-1:]


def wilson(successes,total):
    if total==0: return None
    z=1.959963984540054
    p=successes/total; denominator=1+z*z/total
    center=(p+z*z/(2*total))/denominator
    radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0.0,center-radius),min(1.0,center+radius)]


def summarize(records):
    n=len(records)
    if n==0: raise ValueError("Cannot score an empty evaluation")
    success=sum(r["verified"] for r in records)
    return {"n":n,"valid_rate":sum(r["contract_valid"] for r in records)/n,
        "json_valid_rate":sum(r["json_valid"] for r in records)/n,
        "verified_worlds":success,"verified_rate":success/n,"verified_wilson95":wilson(success,n),
        "contradicted_worlds":sum(r["contradicted"] for r in records),
        "unresolved_worlds":sum(not r["verified"] and not r["contradicted"] for r in records),
        "rejected_evidence_outputs":sum(r["error_category"]=="evidence" for r in records),
        "unbacked_verified_claims":sum(r["unbacked_verified_claims"] for r in records),
        "mean_tool_actions":statistics.mean(r["tool_actions"] for r in records),
        "median_seconds":statistics.median(r["seconds"] for r in records),
        "families":{family:{"n":len(group),"verified_rate":sum(r["verified"] for r in group)/len(group),
                             "verified_wilson95":wilson(sum(r["verified"] for r in group),len(group))}
                    for family in sorted({r["family"] for r in records})
                    for group in [[r for r in records if r["family"]==family]]}}


def paired_comparison(initial,trained):
    before={r["id"]:r for r in initial};after={r["id"]:r for r in trained}
    if set(before)!=set(after): raise ValueError("Paired evaluation worlds differ")
    gained=sum(not before[k]["verified"] and after[k]["verified"] for k in before)
    lost=sum(before[k]["verified"] and not after[k]["verified"] for k in before)
    return {"n":len(before),"gained_worlds":gained,"lost_worlds":lost,
            "verified_rate_change":(gained-lost)/len(before)}


def run_backend(rows,researcher,parent,backend,split,inputs):
    records=[]
    for row in rows:
        started=time.perf_counter()
        with Run(parent.path/"cases",f"{backend}-{split}",{"world":row["id"],"backend":backend,"split":split,
                 "max_actions":8,"judge":"verification-first-rule-v1"},inputs) as child:
            state=Controller(researcher=researcher,max_actions=8).run(row["case"],child)
            trace=getattr(researcher,"last_trace",None)
            if trace is not None: write_json(child.path/"researcher-output.json",trace)
            valid=trace["valid"] if trace is not None else len(state.hypotheses)==1
            json_valid=bool(trace and trace.get("raw_output") is not None and trace.get("error_category")!="syntax") if trace is not None else True
            unbacked=0
            for claim in state.claims:
                if claim.status==Status.VERIFIED:
                    checks=[v for v in state.verifications if v.id in claim.verification_ids]
                    if len(checks)!=2 or any(v.outcome!=Outcome.PASS for v in checks): unbacked+=1
            claim=state.claims[0] if state.claims else None
            truth=row["case"]["experiment"]["observations"][0][1]
            record={"id":row["id"],"group":row["group"],"family":row["family"],"split":split,
                "contract_valid":valid,"json_valid":json_valid,"verified":any(c.status==Status.VERIFIED for c in state.claims),
                "contradicted":any(c.status==Status.CONTRADICTED for c in state.claims),
                "error_category":trace.get("error_category") if trace else None,
                "raw_output":trace.get("raw_output") if trace else None,
                "unbacked_verified_claims":unbacked,"predicted":claim.predicted if claim else None,
                "absolute_prediction_error":abs(claim.predicted-truth) if claim else None,
                "tool_actions":len(state.results),"seconds":time.perf_counter()-started,
                "unknowns":state.unknowns,"run":str(child.path.resolve())}
            write_json(child.path/"evaluation.json",record)
            records.append(record)
    return records


def evaluate(selection_path,holdout_path,runs):
    selection_path,holdout_path=Path(selection_path),Path(holdout_path)
    with Run(Path(runs),"research-holdout",{"selection_sha256":file_hash(selection_path)},[selection_path,holdout_path]) as run:
        selection=json.loads(selection_path.read_text())
        if selection.get("version")!=VERSION or file_hash(holdout_path)!=selection["holdout_sha256"]:
            raise ContractError("Holdout or selection version/hash mismatch")
        holdout=json.loads(holdout_path.read_text())
        if holdout.get("schema")!="aim-research-holdout-v1" or holdout.get("version")!=VERSION:
            raise ContractError("Not the declared research holdout")
        for seed in selection["selections"]:
            for name in ("initial","selected"):
                checkpoint=seed[name]
                if file_hash(checkpoint["checkpoint"])!=checkpoint["sha256"]:
                    raise ContractError("Frozen checkpoint changed")
        # Freeze selection hashes before any case is scored in this evaluator.
        write_json(run.path/"selection-used.json",selection)
        run.event("FROZEN_SELECTION_ACCEPTED",{"sha256":file_hash(selection_path)})
        import torch
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        raw,summary={},{}
        inputs=[selection_path,holdout_path]
        for split in ("test","ood"):
            label="reference-"+split
            raw[label]=run_backend(holdout[split],SingleReferenceResearcher(),run,"reference",split,inputs)
            summary[label]=summarize(raw[label])
            print(json.dumps({"backend":label,"metrics":summary[label]}),flush=True)
        gates={};paired={}
        for seed in selection["selections"]:
            seed_id=seed["seed"]
            for phase in ("initial","selected"):
                checkpoint=seed[phase]["checkpoint"]
                researcher=TransformerResearcher(checkpoint,max_new_tokens=selection["configuration"]["max_new_tokens"])
                for split in ("test","ood"):
                    label=f"seed{seed_id}-{phase}-{split}"
                    raw[label]=run_backend(holdout[split],researcher,run,f"seed{seed_id}-{phase}",split,inputs+[Path(checkpoint)])
                    summary[label]=summarize(raw[label])
                    print(json.dumps({"backend":label,"metrics":summary[label]}),flush=True)
            pair=paired_comparison(raw[f"seed{seed_id}-initial-test"],raw[f"seed{seed_id}-selected-test"])
            paired[str(seed_id)]=pair
            score=summary[f"seed{seed_id}-selected-test"]
            gates[str(seed_id)]={"validity":score["valid_rate"]>=0.9,"verified_success":score["verified_rate"]>=0.25,
                                  "improvement_over_initial":pair["verified_rate_change"]>=0.1}
        unbacked=sum(m["unbacked_verified_claims"] for m in summary.values())
        rates=[summary[f"seed{s['seed']}-selected-test"]["verified_rate"] for s in selection["selections"]]
        report={"version":VERSION,"selection_sha256":file_hash(selection_path),"holdout_sha256":file_hash(holdout_path),
            "backends":summary,"paired":paired,"gates":gates,"unbacked_verified_claims":unbacked,
            "gate_passed":unbacked==0 and all(all(g.values()) for g in gates.values()),
            "seed_dispersion":{"mean_verified_rate":statistics.mean(rates),"min":min(rates),"max":max(rates),
                               "sample_std":statistics.stdev(rates) if len(rates)>1 else None},
            "scope":"fixed synthetic holdout; seeds share worlds and are not independent task replications"}
        write_json(run.path/"case-results.json",raw)
        write_json(run.path/"metrics.json",report)
    return run.path,report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--selection",type=Path,required=True)
    parser.add_argument("--holdout",type=Path,required=True)
    parser.add_argument("--runs",type=Path,default=Path("runs"))
    args=parser.parse_args()
    path,report=evaluate(args.selection,args.holdout,args.runs)
    print(json.dumps({"artifacts":str(path),"gate_passed":report["gate_passed"]}))


if __name__=="__main__": main()

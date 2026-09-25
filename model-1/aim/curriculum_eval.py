"""Frozen curriculum evaluation followed by a separate read-only source audit."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .comparison_eval import evaluate as compare
from .contracts import Evidence, Hypothesis, Outcome
from .curriculum_tasks import EXPERIMENT
from .memory import Memory
from .observation_verifier import ObservationConsistencyVerifier
from .tracking import Run, file_hash, write_json


def audit_observations(case_results,runs):
    case_results=Path(case_results)
    with Run(Path(runs),"observation-consistency-audit",{"version":"1"},[case_results]) as run:
        cases=json.loads(case_results.read_text());audited={};summary={}
        verifier=ObservationConsistencyVerifier()
        for backend,records in cases.items():
            checked=[]
            for record in records:
                source=Path(record["run"]);state_path=source/"state.json"
                state_hash=file_hash(state_path);state=json.loads(state_path.read_text())
                database=source/"memory/memory.sqlite";database_hash=file_hash(database)
                evidence=[Evidence(**e) for e in state["evidence"]]
                checks=[]
                memory=Memory(source/"memory",read_only=True)
                try:
                    for h in state["hypotheses"]:
                        hypothesis=Hypothesis(h["id"],tuple(h["coefficients"]),tuple(h["evidence_ids"]),h["rationale"])
                        checks.append(asdict(verifier.verify(hypothesis,evidence,memory,state["topic"])))
                finally: memory.close()
                if file_hash(state_path)!=state_hash or file_hash(database)!=database_hash:
                    raise RuntimeError("Observation audit changed original state or memory")
                passes=bool(checks) and all(c["outcome"]==Outcome.PASS for c in checks)
                checked.append({"id":record["id"],"source_run":str(source),"state_sha256":state_hash,
                    "memory_sha256":database_hash,
                    "checks":checks,"observation_consistent":passes,"target_verified":record["verified"],
                    "target_and_observations_pass":record["verified"] and passes})
            audited[backend]=checked;n=len(checked)
            summary[backend]={"n":n,"observation_passes":sum(r["observation_consistent"] for r in checked),
                "target_and_observations_pass":sum(r["target_and_observations_pass"] for r in checked),
                "target_verified_without_observation_pass":sum(r["target_verified"] and not r["observation_consistent"] for r in checked),
                "no_hypothesis":sum(not r["checks"] for r in checked)}
        write_json(run.path/"case-checks.json",audited)
        write_json(run.path/"metrics.json",{"backends":summary,
            "scope":"Separate diagnostic; original target claims and gate results remain unchanged"})
    return run.path


def evaluate(selection,holdout,runs):
    compared,metrics=compare(selection,holdout,runs,experiment=EXPERIMENT,arms=("worked","curriculum"),
                             holdout_schema="aim-curriculum-holdout-v1")
    audited=audit_observations(compared/"case-results.json",runs)
    return compared,audited,metrics


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--selection",type=Path,required=True)
    parser.add_argument("--holdout",type=Path,required=True);parser.add_argument("--runs",type=Path,default=Path("runs"))
    args=parser.parse_args();compared,audited,metrics=evaluate(args.selection,args.holdout,args.runs)
    print(json.dumps({"evaluation":str(compared),"observation_audit":str(audited),"arm_gate_passed":metrics["arm_gate_passed"]}))

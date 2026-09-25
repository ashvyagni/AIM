"""Fresh expanded worlds, with trainer-safe metadata and a separate label audit."""
import argparse
import itertools
from pathlib import Path

from .comparison_data import partitions as previous_partitions
from .curriculum_tasks import EXPERIMENT, auxiliary_rows, diagnostic_pools
from .finite_difference import CONTRACT, worked_response
from .memory import Memory
from .research_data import make_record, world_partitions
from .tracking import Run, digest, file_hash, write_json


def partitions():
    excluded={tuple(c) for rows in world_partitions().values() for _,c,_ in rows}
    previous,_=previous_partitions()
    excluded.update(tuple(c) for rows in previous.values() for _,c,_ in rows)
    linear=[];quadratic=[];cubic=[]
    for coefs in itertools.product(range(-19,20),range(-19,20),range(-3,4)):
        if coefs not in excluded:
            (linear if coefs[2]==0 else quadratic).append((digest([EXPERIMENT,list(coefs)]),coefs,"linear" if coefs[2]==0 else "quadratic"))
        for c3 in (-2,-1,1,2):
            values=coefs+(c3,)
            if values not in excluded: cubic.append((digest([EXPERIMENT,list(values)]),values,"cubic"))
    splits={s:[] for s in ("train","validation","test")}
    for candidates,counts in ((sorted(linear),(128,16,16)),(sorted(quadratic),(384,48,48))):
        if len(candidates)<sum(counts): raise ValueError("Insufficient fresh stratum")
        offset=0
        for split,count in zip(splits,counts):
            splits[split].extend(candidates[offset:offset+count]);offset+=count
    splits={s:sorted(rows) for s,rows in splits.items()};splits["ood"]=sorted(cubic)[:64]
    worlds=[tuple(c) for rows in splits.values() for _,c,_ in rows]
    if len(set(worlds))!=len(worlds) or set(worlds)&excluded: raise ValueError("World leakage")
    return splits,sorted(excluded)


def build_dataset(runs):
    with Run(Path(runs),"curriculum-dataset",{"generator":EXPERIMENT}) as run:
        groups,excluded=partitions();memory=Memory(run.path/"source-memory")
        try:
            splits={s:[make_record(*world,s,memory,version=EXPERIMENT) for world in worlds] for s,worlds in groups.items()}
        finally: memory.close()
        for split in ("train","validation"):
            for row in splits[split]:
                row["research_contract"]=CONTRACT
                row["response"]=worked_response(row["world_coefficients"])
                row["auxiliary"]=auxiliary_rows(row)
        train=run.path/"train-validation.json";holdout=run.path/"holdout.json"
        training={"schema":"aim-curriculum-sft-v1","version":EXPERIMENT,
                  "train":splits["train"],"validation":splits["validation"]}
        write_json(train,training)
        write_json(holdout,{"schema":"aim-curriculum-holdout-v1","version":EXPERIMENT,
                           "test":splits["test"],"ood":splits["ood"]})
        prompts={s:{r["prompt"] for r in rows} for s,rows in splits.items()}
        if prompts["train"]&(prompts["validation"]|prompts["test"]): raise ValueError("Research prompt leakage")
        write_json(run.path/"split-audit.json",{"version":EXPERIMENT,"excluded_coefficients":excluded,
            "world_coefficients":{s:[r["world_coefficients"] for r in rows] for s,rows in splits.items()}})
        write_json(run.path/"split-manifest.json",{"version":EXPERIMENT,"excluded_sha256":digest(excluded),
            "excluded_count":len(excluded),"counts":{s:len(rows) for s,rows in splits.items()},
            "group_hashes":{s:digest(sorted(r["group"] for r in rows)) for s,rows in splits.items()},
            "train_validation_sha256":file_hash(train),"holdout_sha256":file_hash(holdout),
            "ood_prompt_overlap_with_train":len(prompts["ood"]&prompts["train"]),"cross_version_world_overlap":0})
        _,counts=diagnostic_pools(training)
        write_json(run.path/"auxiliary-diagnostics-manifest.json",counts)
    return run.path


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--runs",type=Path,default=Path("runs"))
    print(build_dataset(parser.parse_args().runs))

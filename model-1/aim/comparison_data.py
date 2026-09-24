"""Fresh coefficient worlds shared by plain and worked-supervision arms."""
import argparse
import itertools
from pathlib import Path

from .finite_difference import EXPERIMENT
from .memory import Memory
from .research_data import make_record, world_partitions
from .tracking import Run, digest, file_hash, write_json


def partitions():
    excluded={tuple(c) for rows in world_partitions().values() for _,c,_ in rows}
    linear=[];quadratic=[];cubic=[]
    for coefs in itertools.product(range(-9,10),range(-9,10),range(-3,4)):
        if coefs not in excluded:
            (linear if coefs[2]==0 else quadratic).append((digest([EXPERIMENT,list(coefs)]),coefs,"linear" if coefs[2]==0 else "quadratic"))
        for c3 in (-2,-1,1,2):
            values=coefs+(c3,)
            if values not in excluded: cubic.append((digest([EXPERIMENT,list(values)]),values,"cubic"))
    splits={s:[] for s in ("train","validation","test")}
    for candidates,counts in ((sorted(linear),(128,16,16)),(sorted(quadratic),(384,48,48))):
        if len(candidates)<sum(counts): raise ValueError("Insufficient fresh world stratum")
        offset=0
        for split,count in zip(splits,counts):
            splits[split].extend(candidates[offset:offset+count]);offset+=count
    splits={s:sorted(rows) for s,rows in splits.items()}
    splits["ood"]=sorted(cubic)[:64]
    worlds=[tuple(c) for rows in splits.values() for _,c,_ in rows]
    if len(set(worlds))!=len(worlds) or set(worlds)&excluded: raise ValueError("World leakage")
    return splits,sorted(excluded)


def build_dataset(runs):
    with Run(Path(runs),"comparison-dataset",{"generator":EXPERIMENT}) as run:
        groups,excluded=partitions()
        memory=Memory(run.path/"source-memory")
        try:
            splits={s:[make_record(*world,s,memory,version=EXPERIMENT) for world in worlds] for s,worlds in groups.items()}
        finally: memory.close()
        train=run.path/"train-validation.json";holdout=run.path/"holdout.json"
        write_json(train,{"schema":"aim-research-comparison-sft-v1","version":EXPERIMENT,
                          "train":splits["train"],"validation":splits["validation"]})
        write_json(holdout,{"schema":"aim-research-comparison-holdout-v1","version":EXPERIMENT,
                            "test":splits["test"],"ood":splits["ood"]})
        prompts={s:{r["prompt"] for r in rows} for s,rows in splits.items()}
        if prompts["train"]&(prompts["validation"]|prompts["test"]): raise ValueError("In-family prompt overlap")
        # Label-bearing membership is an audit artifact, never trainer metadata.
        write_json(run.path/"split-audit.json",{"version":EXPERIMENT,
            "world_coefficients":{s:[r["world_coefficients"] for r in rows] for s,rows in splits.items()}})
        write_json(run.path/"split-manifest.json",{"version":EXPERIMENT,"excluded_coefficients":excluded,
            "excluded_sha256":digest(excluded),"counts":{s:len(rows) for s,rows in splits.items()},
            "group_hashes":{s:digest(sorted(r["group"] for r in rows)) for s,rows in splits.items()},
            "train_validation_sha256":file_hash(train),"holdout_sha256":file_hash(holdout),
            "ood_prompt_overlap_with_train":len(prompts["ood"]&prompts["train"]),
            "cross_version_world_overlap":0})
    return run.path


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--runs",type=Path,default=Path("runs"))
    print(build_dataset(parser.parse_args().runs))

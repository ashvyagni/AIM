"""Deterministic world-disjoint research traces; no holdout labels enter training."""
import itertools
import json
from pathlib import Path

from .contracts import ResearchState
from .datasets import polynomial_case
from .memory import Memory
from .research_format import VERSION, prompt_for, target_response
from .tracking import Run, digest, file_hash, write_json


def world_id(coefficients):
    return digest([VERSION,list(coefficients)])


def world_partitions():
    pools={s:{"linear":[],"quadratic":[]} for s in ("train","validation","test")}
    for coefs in itertools.product(range(-9,10),range(-9,10),range(-3,4)):
        group=world_id(coefs)
        bucket=int(group[:8],16)%10
        split="train" if bucket<6 else "validation" if bucket<8 else "test"
        pools[split]["linear" if coefs[2]==0 else "quadratic"].append((group,coefs))
    selected={}
    for split,counts in (("train",(128,384)),("validation",(16,48)),("test",(16,48))):
        values=[]
        for family,count in zip(("linear","quadratic"),counts):
            candidates=sorted(pools[split][family])
            if len(candidates)<count:
                raise ValueError("Declared world stratum is too small")
            values.extend((group,coefs,family) for group,coefs in candidates[:count])
        selected[split]=sorted(values)
    ood=[]
    for coefs in itertools.product(range(-9,10),range(-9,10),range(-3,4),(-2,-1,1,2)):
        ood.append((world_id(coefs),coefs,"cubic"))
    selected["ood"]=sorted(ood)[:64]
    groups=[{r[0] for r in rows} for rows in selected.values()]
    if any(a&b for i,a in enumerate(groups) for b in groups[i+1:]):
        raise ValueError("World partitions overlap")
    return selected


def make_record(group,coefficients,family,split,memory,*,version=VERSION):
    target=3+int(group[8:16],16)%3
    case=polynomial_case("p2a-"+group[:20],coefficients,target=target)
    state=ResearchState("1",case["question"],case["topic"],target)
    for source in case["sources"]:
        source["version"]=version
        sid=memory.ingest(**source)
        ev=memory.span(sid,0,len(source["text"]))
        state.evidence.append(ev)
        state.observations.extend(json.loads(source["text"])["observations"])
    prompt,aliases=prompt_for(state)
    return {"id":"world-"+group[:24],"group":group,"split":split,"family":family,
            "generator":version,"label_origin":"programmatic-known-world; not human feedback",
            "rights":"project-generated-fixture","world_coefficients":list(coefficients),
            "prompt":prompt,"response":target_response(coefficients) if split!="ood" else None,
            "aliases":aliases,"source_hashes":[e.source_hash for e in state.evidence],"case":case}


def build_dataset(runs):
    with Run(Path(runs),"research-dataset",{"generator":VERSION,"partition":"coefficient-world-hash"}) as run:
        memory=Memory(run.path/"source-memory")
        try:
            splits={s:[make_record(*world,s,memory) for world in worlds]
                    for s,worlds in world_partitions().items()}
        finally:
            memory.close()
        # The trainer opens only this file; test/OOD labels are a separate artifact.
        train_path=run.path/"train-validation.json"
        holdout_path=run.path/"holdout.json"
        write_json(train_path,{"schema":"aim-research-sft-v1","version":VERSION,
                               "train":splits["train"],"validation":splits["validation"]})
        write_json(holdout_path,{"schema":"aim-research-holdout-v1","version":VERSION,
                                 "test":splits["test"],"ood":splits["ood"]})
        prompts={s:{r["prompt"] for r in rows} for s,rows in splits.items()}
        if prompts["train"] & (prompts["validation"]|prompts["test"]):
            raise ValueError("In-family prompt leakage")
        write_json(run.path/"split-manifest.json",{
            "version":VERSION,"counts":{s:len(rows) for s,rows in splits.items()},
            "strata":{s:{f:sum(r["family"]==f for r in rows) for f in ("linear","quadratic","cubic")} for s,rows in splits.items()},
            "group_hashes":{s:digest(sorted(r["group"] for r in rows)) for s,rows in splits.items()},
            "train_validation_sha256":file_hash(train_path),"holdout_sha256":file_hash(holdout_path),
            "ood_prompt_overlap_with_train":len(prompts["ood"]&prompts["train"]),
            "ood_note":"Different cubic worlds can share the first three observations with a quadratic; such ambiguity is not removed post hoc"})
    return run.path


def load_training_data(path):
    record=json.loads(Path(path).read_text())
    if record.get("schema")!="aim-research-sft-v1" or record.get("version")!=VERSION:
        raise ValueError("Unsupported structured research dataset")
    if set(record)!={"schema","version","train","validation"}:
        raise ValueError("Training file must contain train/validation only")
    for split in ("train","validation"):
        rows=record[split]
        if not isinstance(rows,list) or not rows:
            raise ValueError("Empty research split")
        if any(r["split"]!=split or not isinstance(r["response"],str) for r in rows):
            raise ValueError("Research row split/label mismatch")
    if {r["group"] for r in record["train"]}&{r["group"] for r in record["validation"]}:
        raise ValueError("Training and validation worlds overlap")
    return record


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--runs",type=Path,default=Path("runs"))
    print(build_dataset(parser.parse_args().runs))

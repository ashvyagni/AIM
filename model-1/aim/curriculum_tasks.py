"""Versioned auxiliary arithmetic tasks; labels never enter research prompts."""
import json
from fractions import Fraction

from .contracts import ContractError, finite_number
from .research_format import _unique_object, _invalid_constant
from .tracking import digest

EXPERIMENT = "aim-arithmetic-curriculum-v1"
KINDS = ("subtract", "second_difference", "reconstruct")


def auxiliary_rows(row):
    points=json.loads(row["prompt"])["evidence"]["E0"]
    if [p[0] for p in points]!=[0,1,2]: raise ContractError("Auxiliary teachers require observations at 0,1,2")
    values=[finite_number(p[1]) for p in points]
    if any(not v.is_integer() for v in values): raise ContractError("Curriculum fixture requires integer observations")
    y0,y1,y2=map(int,values);d1=y1-y0;d2=y2-2*y1+y0
    if d2%2: raise ContractError("Curriculum reconstruction requires integral quadratic coefficients")
    examples=[("subtract",{"task":"subtract","a":y1,"b":y0},{"value":d1}),
              ("subtract",{"task":"subtract","a":y2,"b":y1},{"value":y2-y1}),
              ("second_difference",{"task":"second-difference","y":[y0,y1,y2]},{"value":d2}),
              ("reconstruct",{"task":"coefficients","y0":y0,"d1":d1,"d2":d2},
               {"coefficients":[[y0,d1-d2//2,d2//2]]})]
    return [{"kind":kind,"prompt":json.dumps(prompt,separators=(",",":"))+"\n",
             "response":json.dumps(response,separators=(",",":")),"world_group":row["group"],
             "split":row["split"],"label_origin":"programmatic observation arithmetic; not human feedback"}
            for kind,prompt,response in examples]


def batch_for(config,rows,step):
    if config["arm"]=="worked": return rows,["research"]*len(rows)
    if config["arm"]!="curriculum": raise ContractError("Unknown curriculum arm")
    if step<=600: count,kind,index=len(rows)//2,"subtract",0 if step%2 else 1
    elif step<=1200: count,kind,index=len(rows)//4,"second_difference",0
    else: count,kind,index=len(rows)//4,"reconstruct",0
    batch=list(rows);roles=["research"]*len(rows)
    for i in range(count):
        candidates=[r for r in rows[i]["auxiliary"] if r["kind"]==kind]
        batch[i]=candidates[index];roles[i]=kind
    return batch,roles


def diagnostic_pools(data):
    all_rows={split:[x for row in data[split] for x in row["auxiliary"]] for split in ("train","validation")}
    training_prompts={r["prompt"] for r in all_rows["train"]}
    pools={};counts={}
    for split in ("train","validation"):
        for kind in KINDS:
            unique={r["prompt"]:r for r in all_rows[split] if r["kind"]==kind}
            eligible={p:r for p,r in unique.items() if split=="train" or p not in training_prompts}
            key=f"{split}-{kind}"
            pools[key]=[eligible[p] for p in sorted(eligible,key=digest)[:32]]
            counts[key]={"unique":len(unique),"excluded_training_overlap":len(unique)-len(eligible),
                         "eligible":len(eligible),"sampled":len(pools[key])}
    return pools,counts


def score_auxiliary(text,row):
    try:
        actual=json.loads(text,object_pairs_hook=_unique_object,parse_constant=_invalid_constant)
        expected=json.loads(row["response"])
        if not isinstance(actual,dict) or set(actual)!=set(expected): raise ContractError("Auxiliary schema mismatch")
        if row["kind"]=="reconstruct":
            values=actual["coefficients"]
            if not isinstance(values,list) or len(values)!=1 or not isinstance(values[0],list) or len(values[0])!=3:
                raise ContractError("Expected one coefficient triple")
            values=values[0];truth=expected["coefficients"][0]
        else: values=[actual["value"]];truth=[expected["value"]]
        for value in values: finite_number(value,1e6)
        return {"valid":True,"correct":all(Fraction(str(a))==Fraction(str(b)) for a,b in zip(values,truth))}
    except (ValueError,TypeError,KeyError):
        return {"valid":False,"correct":False}

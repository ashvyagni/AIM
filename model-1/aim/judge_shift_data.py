"""Fresh ambiguity-grouped fixtures with source-backed, independently checked labels."""
import itertools
from dataclasses import asdict
from pathlib import Path

from .contracts import Claim, Outcome, ResearchState, ToolResult
from .datasets import world_value
from .judge_shift_features import VERSION, extract, payload
from .memory import Memory
from .researcher import PolynomialResearcher, predict
from .tracking import Run, canonical, digest, file_hash, write_json
from .verifiers import MeasurementVerifier


def groups():
    result = {s: [] for s in ("train", "calibration", "test", "ood")}
    for linear in (True, False):
        candidates = sorted((digest([VERSION, c]), c) for c in
            itertools.product(range(30,91), range(-20,21), range(-3,4)) if (c[2] == 0) == linear)
        offset = 0
        for split, count in zip(result, (64,32,32,32)):
            for index, (group, coefs) in enumerate(candidates[offset:offset+count]):
                result[split].append((group, coefs, 3 + index % 2))
            offset += count
    return {s: sorted(rows) for s,rows in result.items()}


def variants(q, group, split):
    a,b,c = q
    if split == "ood":
        return [(q, "base")] + [((a,b+2*k,c-k,-2*k,k), "quartic") for k in (-2,1,2)]
    k = (-1,1)[int(group[:8],16) % 2]
    return [(q, "base"), ((a,b+2*k,c-3*k,k), "cubic")]


def build_dataset(runs):
    with Run(Path(runs), "judge-shift-data", {"version": VERSION}) as run:
        splits = {}; audit = []; memory = Memory(run.path/"memory")
        try:
            for split, bases in groups().items():
                rows = []
                for group, q, n in bases:
                    target = (4,5,6)[int(group[8:16],16) % 3]
                    for index, (coefs,family) in enumerate(variants(q,group,split)):
                        wid = digest([VERSION, list(coefs)])
                        topic = "instrument-" + wid
                        points = [[x,world_value(coefs,x)] for x in range(n)]
                        text = canonical({"topic": topic, "observations": points})
                        sid = memory.ingest(text, uri=f"aim://judge-shift/{wid}/observations",
                            version=VERSION, rights="project-generated-fixture", title=topic)
                        ev = memory.span(sid,0,len(text))
                        state = ResearchState("1", "Forecast next measurement", topic, target,
                            observations=points, evidence=[ev])
                        # Candidate generation has no hidden coefficients or future value.
                        candidates = PolynomialResearcher().hypothesize(state)
                        value = world_value(coefs,target)
                        measured = ToolResult("measurement-"+wid, Outcome.PASS, value,
                            "Exact rational synthetic environment", 0.0, VERSION)
                        measured_text = canonical({"x":target,"value":value})
                        msid = memory.ingest(measured_text,uri=f"aim://judge-shift/{wid}/measurement",
                            version=VERSION,rights="project-generated-fixture",title="Future measurement",retrievable=False)
                        mev = memory.span(msid,0,len(measured_text))
                        for h in candidates:
                            rid = digest([wid,h.id])
                            pre = payload(state,h)
                            fs = {mode:extract(pre,mode) for mode in ("five","rich")}
                            claim = Claim(rid,h.id,"Scoped next measurement",predict(h.coefficients,target),target,[ev.id])
                            check = MeasurementVerifier().verify(claim,measured,mev)
                            if check.outcome not in (Outcome.PASS,Outcome.FAIL):
                                raise ValueError("Unresolved labels may not be coerced to false")
                            rows.append({"id":rid,"group":group,"world":wid,"split":split,
                                "family":family,"observations_count":n,"input":pre,"features":fs,
                                "label":int(check.outcome==Outcome.PASS),"label_origin":"programmatic-verifier",
                                "source":asdict(ev),"verification_id":check.id})
                            audit.append({"id":rid,"coefficients":list(coefs),"base":list(q),
                                "claim":asdict(claim),"measurement":asdict(measured),"measurement_evidence":asdict(mev),
                                "verification":asdict(check)})
                splits[split] = rows
        finally:
            memory.close()
        write_json(run.path/"train-calibration.json",{"version":VERSION,"train":splits["train"],"calibration":splits["calibration"]})
        write_json(run.path/"holdout.json",{"version":VERSION,"test":splits["test"],"ood":splits["ood"]})
        write_json(run.path/"label-audit.json",{"version":VERSION,"rows":audit})
        write_json(run.path/"manifest-data.json",{"version":VERSION,"counts":{s:len(v) for s,v in splits.items()},
            "group_counts":{s:len({r['group'] for r in rows}) for s,rows in splits.items()},
            "train_sha256":file_hash(run.path/"train-calibration.json"),"holdout_sha256":file_hash(run.path/"holdout.json")})
    return run.path

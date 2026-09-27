"""Train/validation-only inputs for separately versioned language objectives."""
from .corpus import read_json, require
from .datasets import VERSION, arithmetic_data, assert_disjoint
from .language_contract import output_contract
from .tracking import digest


def prepare(task, stage, path=None):
    require(task in {"arithmetic", "symbolic", "external", "structured"}, "unknown language task")
    if task in {"external", "structured"}:
        require(path is not None, "this task requires an explicit dataset file")
        require(stage != "rlvr", "external/structured RLVR requires a reviewed verifier adapter")
        record = read_json(path, 10*1024*1024)
        require(isinstance(record, dict) and set(record) == {"schema", "version", "train", "validation"}, "dataset must contain train/validation only")
        if task == "structured":
            from .research_format import VERSION as RESEARCH_VERSION
            require(stage == "sft" and record["schema"] == "aim-research-sft-v1" and record["version"] == RESEARCH_VERSION, "structured path accepts versioned SFT records only")
        else:
            require(record["schema"] == "aim-language-data-v1", "external language dataset schema mismatch")
        version = record["version"]
        splits = {s: record[s] for s in ("train", "validation")}
    else:
        require(path is None, "procedural tasks cannot silently replace their data")
        if task == "symbolic":
            from .symbolic_data import symbolic_data, VERSION as SYMBOLIC_VERSION
            splits = {s: symbolic_data(s) for s in ("train", "validation")}
            version = SYMBOLIC_VERSION
        else:
            splits = {"train": arithmetic_data("train"), "validation": arithmetic_data("validation", 32)}
            version = VERSION
    require(isinstance(version, str) and 0 < len(version) <= 128, "dataset version required")
    ids = set()
    normalized = {}
    for split, rows in splits.items():
        require(isinstance(rows, list) and 1 <= len(rows) <= 10000, "dataset split requires 1..10000 examples")
        normalized[split] = []
        for item in rows:
            require(isinstance(item, dict), "dataset row must be an object")
            row = dict(item)
            for key in ("id", "group", "prompt", "response", "rights", "label_origin"):
                require(isinstance(row.get(key), str) and 0 < len(row[key]) <= 262144, "missing/big language row field: "+key)
            require(row["id"] not in ids, "duplicate language row ID")
            ids.add(row["id"])
            require(row.get("split", split) == split, "row belongs to a different split")
            row["split"] = split
            if task == "external":
                allowed = {"id", "group", "prompt", "response", "rights", "label_origin", "split", "annotation_batch", "chosen", "rejected"}
                require(set(row) <= allowed, "undeclared external row metadata; holdout labels are not training metadata")
                require(row["label_origin"] in {"human", "synthetic", "programmatic"}, "external label origin must be explicit")
                if row["label_origin"] == "human":
                    require(isinstance(row.get("annotation_batch"), str) and bool(row["annotation_batch"].strip()), "human annotations require a batch reference")
            if stage == "preference" or "chosen" in row or "rejected" in row:
                require(all(isinstance(row.get(k), str) and 0 < len(row[k]) <= 262144 for k in ("chosen", "rejected")), "preference pair is incomplete")
                require(row["chosen"] != row["rejected"], "preference responses must differ")
            if stage == "rlvr" and task == "arithmetic":
                n = sum(row["operands"])
                row["candidates"] = [str(n-1), str(n), str(n+1)]
            if stage == "rlvr":
                require(isinstance(row.get("candidates"), list) and 2 <= len(row["candidates"]) <= 16 and
                        all(isinstance(x, str) and x for x in row["candidates"]), "bounded verifier candidates required")
                require(len(set(row["candidates"])) == len(row["candidates"]), "duplicate verifier candidate")
            normalized[split].append(row)
    assert_disjoint(*normalized.values())
    require(not ({r["prompt"] for r in normalized["train"]} & {r["prompt"] for r in normalized["validation"]}), "exact prompt leakage across language splits")
    content = {"schema": "aim-prepared-language-data-v1", "version": version, "task": task, "stage": stage,
               "output_contract": output_contract(task), **normalized}
    return {**content, "dataset_hash": digest(content)}


def rewards(task, row):
    require(task in {"arithmetic", "symbolic"}, "no verifier registered for this task")
    if task == "symbolic":
        from .symbolic_data import symbolic_reward
        verifier = symbolic_reward
    else:
        from .datasets import arithmetic_reward
        verifier = arithmetic_reward
    values = [verifier(row, answer) for answer in row["candidates"]]
    require(all(value in (0., 1.) for value in values), "verifier must return defined binary outcomes")
    return values

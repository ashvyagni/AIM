"""Versioned procedural fixtures. They are engineering data, not human preferences."""
from __future__ import annotations

import json
import random
from fractions import Fraction

from .contracts import ContractError, Evidence, ResearchState
from .judge import features
from .researcher import PolynomialResearcher
from .tracking import digest

VERSION = "aim-procedural-v1"


def arithmetic_data(split="train", count=64):
    # Partition complete operand pairs deterministically, not individual rows.
    rows = []
    for a in range(30):
        for b in range(30):
            bucket = int(digest([a, b])[:8], 16) % 10
            selected = bucket < 6 if split == "train" else 6 <= bucket < 8 if split == "validation" else bucket >= 8
            if selected:
                rows.append({"id": f"add-{a}-{b}", "group": f"operands-{a}-{b}", "split": split,
                             "prompt": f"{a}+{b}=", "response": str(a+b), "chosen": str(a+b),
                             "rejected": str(a+b+1), "operands": [a,b], "operation": "add",
                             "label_origin": "procedural-exact-arithmetic; not human feedback",
                             "generator": VERSION, "rights": "project-generated-fixture"})
    if len(rows) < count:
        raise ValueError("Requested fixture count exceeds available distinct operand pairs")
    return rows[:count]


def arithmetic_reward(row, answer):
    """Independent exact verifier; does not use the chosen/preference field."""
    try:
        expected = Fraction(row["operands"][0]) + Fraction(row["operands"][1])
        return float(Fraction(answer) == expected)
    except (ValueError, ZeroDivisionError):
        return 0.0


def world_value(coefficients, x):
    # Exact rational environment implementation, separate from floating researcher fit.
    x = Fraction(x)
    value = Fraction(0)
    for i, coefficient in enumerate(coefficients):
        value += Fraction(coefficient) * x ** i
    return float(value)


def polynomial_case(case_id, coefficients, *, target=4, available=True, conflicting=False, observations=3):
    topic = f"instrument-{case_id}"
    rows = [[x, world_value(coefficients, x)] for x in range(observations)]
    sources = [{"title": topic, "uri": f"aim://fixtures/{case_id}/observations", "version": VERSION,
                "rights": "project-generated-fixture", "text": json.dumps({"topic": topic, "observations": rows})}]
    if conflicting:
        sources.append({**sources[0], "uri": sources[0]["uri"] + "/conflict",
                        "text": json.dumps({"topic": topic, "observations": [[0, rows[0][1]+1]]})})
    return {"id": case_id, "question": f"Which candidate predicts {topic} at x={target}?",
            "topic": topic, "target_x": target, "sources": sources,
            "experiment": {"available": available, "observations": [[target, world_value(coefficients,target)]]}}


def judge_data(split, count, seed=17, ood=False):
    rng = random.Random(seed)
    records, seen = [], set()
    while len(records) < count:
        degree = 3 if ood else rng.choice([1, 2])
        coefs = [rng.randint(-8, 8) for _ in range(degree)] + [rng.choice([-5,-3,-1,1,3,5])]
        group = digest(coefs)
        bucket = int(group[:8], 16) % 10
        valid = bucket < 6 if split == "train" else 6 <= bucket < 8 if split == "validation" else bucket >= 8
        if not valid or group in seen:
            continue
        seen.add(group)
        n = rng.choice([3, 4])
        target = rng.choice([4, 5, 6])
        state = ResearchState("1", "Synthetic forecast", "generated", target)
        state.observations = [[x, world_value(coefs, x)] for x in range(n)]
        # Feature construction uses only observed points and candidate coefficients.
        state.evidence = [Evidence("fixture", "fixture", "fixture", 0, 1, "x", "fixture", "fixture")]
        for h in PolynomialResearcher().hypothesize(state):
            expected = Fraction(str(world_value(coefs, target)))
            proposed = sum(Fraction(str(c)) * Fraction(target)**i for i, c in enumerate(h.coefficients))
            label = int(abs(expected - proposed) <= Fraction(1,100_000_000))
            records.append({"features": features(state,h), "label": label, "group": group,
                            "split": split, "family": "cubic-ood" if ood else "linear-quadratic",
                            "generator": VERSION, "target": "pre-measurement numeric agreement"})
            if len(records) >= count:
                break
    return records


def assert_disjoint(*datasets):
    groups = [set(x["group"] for x in rows) for rows in datasets]
    for i, a in enumerate(groups):
        for b in groups[i+1:]:
            if a & b:
                raise ValueError("Dataset groups leak across splits")


def load_supervised_dataset(path):
    """Explicit SFT/preference intake. Declared human labels are never inferred."""
    from pathlib import Path
    path = Path(path)
    if path.stat().st_size > 10_000_000:
        raise ContractError("Miniature dataset file exceeds 10 MB")
    record = json.loads(path.read_text())
    if record.get("schema") != "aim-supervised-v1" or not record.get("version"):
        raise ContractError("Dataset schema and version required")
    for split in ("train", "validation"):
        rows = record.get(split)
        if not isinstance(rows, list) or not 1 <= len(rows) <= 10000:
            raise ContractError("Dataset split must contain 1..10000 rows")
        for row in rows:
            required = ("id", "group", "prompt", "response", "chosen", "rejected", "rights", "label_origin")
            if any(not isinstance(row.get(k), str) or not row[k].strip() for k in required):
                raise ContractError("Dataset row has missing text, grouping or provenance fields")
            if row["label_origin"] not in {"human", "synthetic", "programmatic"}:
                raise ContractError("Declare human, synthetic or programmatic label origin")
            if row["label_origin"] == "human" and not row.get("annotation_batch"):
                raise ContractError("Human labels require an annotation batch identifier")
            if row["chosen"] == row["rejected"]:
                raise ContractError("Chosen and rejected responses must differ")
            if row.get("split", split) != split:
                raise ContractError("Row split conflicts with enclosing split")
            row["split"] = split
    assert_disjoint(record["train"], record["validation"])
    # Reject exact prompt leakage even if a caller assigns inconsistent group IDs.
    if {r["prompt"] for r in record["train"]} & {r["prompt"] for r in record["validation"]}:
        raise ContractError("Exact prompts leak across training and validation")
    ids = [r["id"] for s in ("train", "validation") for r in record[s]]
    if len(ids) != len(set(ids)):
        raise ContractError("Duplicate dataset record IDs")
    return record["train"], record["validation"], record["version"]

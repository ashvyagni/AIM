"""Versioned binomial fixtures; programmatic labels are not human feedback."""
import json
from types import SimpleNamespace

from .contracts import ContractError
from .symbolic import ASSUMPTIONS, SCHEMA, check_identity, identity_request
from .symbolic_loop import parse_symbolic_response, symbolic_prompt

VERSION = "symbolic-binomial-engineering-v1"


def expression(a):
    return f"(x{'+' if a >= 0 else '-'}{abs(a)})**2"


def response(rhs):
    return json.dumps({"rhs": rhs, "evidence_ids": ["E0"]}, separators=(",", ":"))


def symbolic_data(split):
    ranges = {"train": range(-12, 12), "validation": range(12, 20), "test": range(20, 28)}
    if split not in ranges:
        raise ContractError("unsupported symbolic split")
    rows = []
    for a in ranges[split]:
        lhs = expression(a)
        good = response(f"x**2+({2*a})*x+({a*a})")
        bad = response(f"x**2+({2*a})*x+({a*a+1})")
        rows.append({"id": f"square-{a}", "group": f"binomial-offset-{a}", "split": split,
                     "lhs": lhs, "prompt": symbolic_prompt(lhs), "response": good, "chosen": good,
                     "rejected": bad, "candidates": [bad, good, response("x/0")],
                     "label_origin": "programmatic", "rights": "project-generated-fixture", "generator": VERSION})
    return rows


def symbolic_reward(row, text):
    # E0 is the row's expression declaration, not an external literature citation.
    # Do not read response/chosen/rejected as a reward key.
    try:
        candidate = parse_symbolic_response(text, [SimpleNamespace(id="E0")])
        return float(check_identity(identity_request(row["lhs"], candidate.rhs))["outcome"] == "PASS")
    except ContractError:
        return 0.0


def symbolic_case(lhs="(x+3)**2", case_id="symbolic-demo"):
    return {"schema": SCHEMA, "id": case_id, "question": f"Find and check a polynomial expansion of {lhs}.",
            "lhs": lhs, "assumptions": dict(ASSUMPTIONS), "sources": [{"title": "Expression declaration",
            "uri": f"aim://symbolic/{case_id}", "version": VERSION, "rights": "project-generated-fixture",
            "text": json.dumps({"expression": lhs, "assumptions": ASSUMPTIONS})}]}

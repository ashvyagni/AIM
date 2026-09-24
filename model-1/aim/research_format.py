"""Versioned, evidence-bound input/output contract for the Phase 2A experiment."""
import json

from .contracts import ContractError, Hypothesis, finite_number

VERSION = "aim-structured-research-v1"


class ResearchOutputError(ContractError):
    def __init__(self, category, message):
        super().__init__(message)
        self.category = category


def number(value):
    value = finite_number(value)
    return int(value) if value.is_integer() else value


def prompt_for(state):
    """Only observation sources enter context; measurement results never enter it."""
    if state.results or state.verifications:
        # Calculations occur only after hypothesis creation in the present controller.
        raise ContractError("Structured Researcher requires a pre-action state")
    evidence, aliases = {}, {}
    for ev in state.evidence:
        try:
            record = json.loads(ev.quote)
        except ValueError:
            continue
        if not isinstance(record, dict) or record.get("topic") != state.topic:
            continue
        rows = record.get("observations")
        if not isinstance(rows, list) or not rows or len(rows) > 100:
            raise ContractError("Invalid structured observation evidence")
        values = []
        for pair in rows:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ContractError("Expected evidence observation pairs")
            values.append([number(pair[0]), number(pair[1])])
        alias = f"E{len(aliases)}"
        aliases[alias] = ev.id
        evidence[alias] = sorted(values)
    if not aliases:
        raise ContractError("No structured observation evidence for Researcher")
    prompt = json.dumps({"task":"fit-degree<=2","evidence":evidence,"x":number(state.target_x)},
                        separators=(",",":"), allow_nan=False) + "\n"
    return prompt, aliases


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ResearchOutputError("syntax", "Duplicate output JSON key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ResearchOutputError("syntax", f"Nonfinite JSON value: {value}")


def parse_hypothesis(text, aliases):
    try:
        record = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (ValueError, TypeError) as exc:
        if isinstance(exc, ResearchOutputError):
            raise
        raise ResearchOutputError("syntax", "Researcher output is not valid JSON") from exc
    if not isinstance(record, dict) or set(record) != {"coefficients", "evidence"}:
        raise ResearchOutputError("schema", "Expected coefficients and evidence fields only")
    candidates = record["coefficients"]
    if (not isinstance(candidates, list) or len(candidates) != 1 or
            not isinstance(candidates[0], list) or len(candidates[0]) != 3):
        raise ResearchOutputError("schema", "Exactly one three-coefficient candidate is required")
    try:
        coefficients = tuple(finite_number(x, 1e6) for x in candidates[0])
    except ContractError as exc:
        raise ResearchOutputError("schema", str(exc)) from exc
    references = record["evidence"]
    if (not isinstance(references, list) or not references or
            any(not isinstance(r, str) for r in references) or
            len(references) != len(set(references)) or set(references) != set(aliases)):
        raise ResearchOutputError("evidence", "Citations must name each supplied evidence alias exactly once")
    return [Hypothesis("h-neural-0", coefficients, tuple(aliases[r] for r in references),
                       "Native structured Researcher candidate; independent verification required")]


def target_response(coefficients):
    return json.dumps({"coefficients":[list(coefficients)],"evidence":["E0"]},separators=(",",":"),allow_nan=False)

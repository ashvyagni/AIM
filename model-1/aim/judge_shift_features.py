"""Allowlisted pre-measurement inputs; no outcomes or generator metadata."""
import math
from fractions import Fraction

from .contracts import ContractError, Hypothesis, ResearchState, finite_number
from .judge import features

VERSION = "aim-judge-shift-v1"
FEATURE_VERSION = "observed-polynomial-features-v2"


def payload(state, hypothesis):
    return {"observations": state.observations, "coefficients": list(hypothesis.coefficients),
            "target_x": state.target_x}


def extract(record, mode="rich"):
    if not isinstance(record, dict) or set(record) != {"observations", "coefficients", "target_x"}:
        raise ContractError("Judge inputs must contain only observed points, candidate and target coordinate")
    if mode not in {"five", "rich"}:
        raise ContractError("Unknown Judge feature mode")
    target = finite_number(record["target_x"])
    points = record["observations"]
    if not isinstance(points, list) or not 2 <= len(points) <= 100:
        raise ContractError("Invalid observed point count")
    seen = set()
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ContractError("Invalid observation")
        x, _ = [finite_number(v) for v in point]
        if x >= target or x in seen:
            raise ContractError("Duplicate or non-past observation")
        seen.add(x)
    coefficients = record["coefficients"]
    if not isinstance(coefficients, list):
        raise ContractError("Candidate coefficients must be a list")
    h = Hypothesis("feature-candidate", tuple(coefficients), ("observations",), "feature only")
    state = ResearchState("1", "", "", target, observations=points)
    result = features(state, h)
    if mode == "rich":
        coefs = [Fraction(str(c)) for c in coefficients]
        residuals = [abs(sum(c * Fraction(str(x))**i for i,c in enumerate(coefs))-Fraction(str(y)))
                     for x,y in points]
        scale = max([abs(y) for _,y in points]+[1.0])
        result += [float(max(residuals) <= Fraction(1, 100_000_000)),
                   math.log1p(min(float(max(residuals))/scale, 1e6))]
    return result

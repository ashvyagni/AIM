"""Versioned domain contracts. Confidence and verification are distinct objects."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol


class ContractError(ValueError):
    pass


class Status(str, Enum):
    HYPOTHESIS = "HYPOTHESIS"
    VERIFIED = "VERIFIED"
    CONTRADICTED = "CONTRADICTED"
    UNKNOWN = "UNKNOWN"
    UNVERIFIED = "UNVERIFIED"


class Outcome(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"


def finite_number(value: Any, limit: float = 1e9) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError("Expected a finite numeric value")
    if not math.isfinite(value) or abs(value) > limit:
        raise ContractError("Numeric value exceeds finite domain bounds")
    return float(value)


@dataclass(frozen=True)
class Evidence:
    id: str
    source_id: str
    source_hash: str
    start: int
    end: int
    quote: str
    quote_hash: str
    locator: str


@dataclass(frozen=True)
class Hypothesis:
    id: str
    coefficients: tuple[float, ...]  # ascending powers
    evidence_ids: tuple[str, ...]
    rationale: str

    def __post_init__(self):
        if not 1 <= len(self.coefficients) <= 3:
            raise ContractError("Miniature domain supports degree 0..2")
        for x in self.coefficients:
            finite_number(x)
        if not self.id or not self.evidence_ids:
            raise ContractError("Hypotheses require IDs and evidence")


@dataclass(frozen=True)
class Action:
    id: str
    name: str
    arguments: dict[str, Any]
    timeout_seconds: float = 5.0

    def __post_init__(self):
        if not self.id or not self.name:
            raise ContractError("Action identity required")
        if not 0 < finite_number(self.timeout_seconds) <= 30:
            raise ContractError("Action timeout must be in (0,30]")


@dataclass(frozen=True)
class ToolResult:
    action_id: str
    outcome: Outcome
    value: Any
    detail: str
    elapsed_seconds: float
    tool_version: str


@dataclass(frozen=True)
class Verification:
    id: str
    claim_id: str
    claim_hash: str
    verifier: str
    version: str
    outcome: Outcome
    scope: str
    evidence_ids: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class Decision:
    claim_id: str
    probability: float | None
    target: str
    action: str
    model_id: str
    calibration_status: str

    def __post_init__(self):
        if self.probability is not None and not 0 <= finite_number(self.probability) <= 1:
            raise ContractError("Probability outside [0,1]")
        if self.action not in {"VERIFY", "ABSTAIN"}:
            raise ContractError("Unsupported decision action")


@dataclass
class Claim:
    id: str
    hypothesis_id: str
    text: str
    predicted: float
    target_x: float
    evidence_ids: list[str]
    status: Status = Status.HYPOTHESIS
    verification_ids: list[str] = field(default_factory=list)
    forecast: Decision | None = None


@dataclass
class ResearchState:
    schema_version: str
    question: str
    topic: str
    target_x: float
    phase: str = "CREATED"
    plan: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    observations: list[list[float]] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)
    verifications: list[Verification] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    final_response: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Researcher(Protocol):
    model_id: str

    def plan(self, state: ResearchState) -> list[str]: ...

    def hypothesize(self, state: ResearchState) -> list[Hypothesis]: ...


class Judge(Protocol):
    model_id: str

    def decide(self, state: ResearchState, hypothesis: Hypothesis, claim: Claim) -> Decision: ...


class Verifier(Protocol):
    name: str
    version: str

    def verify(self, claim: Claim, measurement: ToolResult, evidence: Evidence | None) -> Verification: ...

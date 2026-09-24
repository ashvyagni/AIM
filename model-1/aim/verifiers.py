"""Scoped verification is independent of model confidence and preferences."""
from fractions import Fraction

from .contracts import Claim, Evidence, Outcome, ToolResult, Verification
from .tracking import digest


def claim_identity(claim: Claim):
    return {"id": claim.id, "text": claim.text, "hypothesis_id": claim.hypothesis_id,
            "predicted": claim.predicted, "target_x": claim.target_x, "evidence_ids": claim.evidence_ids}


class MeasurementVerifier:
    name, version = "numeric-observation-agreement", "1"

    def verify(self, claim: Claim, measurement: ToolResult, evidence: Evidence | None):
        status = Outcome.UNKNOWN
        detail = "No successful measurement; unresolved is not false"
        if measurement.outcome == Outcome.PASS and evidence is not None:
            delta = abs(Fraction(str(claim.predicted)) - Fraction(str(measurement.value)))
            status = Outcome.PASS if delta <= Fraction(1, 100_000_000) else Outcome.FAIL
            detail = f"absolute_difference={float(delta)}; absolute_tolerance=1e-8"
        elif measurement.outcome in {Outcome.ERROR, Outcome.TIMEOUT}:
            status = measurement.outcome
            detail = measurement.detail
        return Verification("v-" + digest([claim_identity(claim), measurement])[:24], claim.id,
                            digest(claim_identity(claim)), self.name, self.version, status,
                            f"Prediction agreement with synthetic measurement at x={claim.target_x}; not a proof of the global law",
                            (evidence.id,) if evidence else (), detail)


class ProvenanceVerifier:
    name, version = "source-span-integrity", "1"

    def verify_spans(self, claim: Claim, memory, evidence: list[Evidence]):
        index = {e.id: e for e in evidence}
        good = bool(claim.evidence_ids) and all(i in index and memory.validate(index[i]) for i in claim.evidence_ids)
        return Verification("vp-" + digest(claim_identity(claim))[:24], claim.id, digest(claim_identity(claim)),
                            self.name, self.version, Outcome.PASS if good else Outcome.FAIL,
                            "Exact source/version/span identity only; no general natural-language entailment",
                            tuple(claim.evidence_ids), "All cited spans resolve and match originals" if good else "Invalid provenance")

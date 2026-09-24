"""Judges issue forecasts BEFORE measurement; they never certify claims."""
from __future__ import annotations

import math

from .contracts import Claim, Decision, Hypothesis, ResearchState
from .researcher import predict

TARGET = "candidate prediction matches next synthetic measurement within absolute tolerance 1e-8"
FEATURE_VERSION = "polynomial-features-v1"


def features(state: ResearchState, hypothesis: Hypothesis):
    points = state.observations
    scale = max([abs(y) for _, y in points] + [1.0])
    residual = sum(abs(predict(hypothesis.coefficients, x) - y) for x, y in points) / max(len(points), 1) / scale
    distance = min([abs(state.target_x - x) for x, _ in points] or [100.0])
    return [len(hypothesis.coefficients) / 3, min(len(points), 8) / 8,
            math.log1p(min(residual, 1e6)), min(distance, 10) / 10,
            min(abs(predict(hypothesis.coefficients, state.target_x)) / scale, 20) / 20]


class VerificationFirstJudge:
    model_id = "verification-first-rule-v1"

    def decide(self, state: ResearchState, hypothesis: Hypothesis, claim: Claim):
        return Decision(claim.id, None, TARGET, "VERIFY", self.model_id, "untrained; no probability claimed")


class CalibratedJudge:
    def __init__(self, checkpoint, abstain_below=0.0):
        from .neural import load_judge
        from .tracking import file_hash
        self.model, self.temperature, self.record = load_judge(checkpoint)
        self.model_id = "aim-judge-" + file_hash(checkpoint)
        if not 0 <= abstain_below <= 1:
            raise ValueError("Invalid abstention threshold")
        self.abstain_below = abstain_below

    def decide(self, state, hypothesis, claim):
        import torch
        with torch.no_grad():
            value = self.model(torch.tensor([features(state, hypothesis)], dtype=torch.float32))
            probability = torch.sigmoid(value / self.temperature).item()
        return Decision(claim.id, probability, TARGET,
                        "VERIFY" if probability >= self.abstain_below else "ABSTAIN", self.model_id,
                        "temperature fit on synthetic validation; scope-limited, see checkpoint metrics")

"""Controller owns state, capability boundaries, evidence updates and final claim status."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict

from .contracts import Action, Claim, ContractError, Outcome, ResearchState, Status, finite_number
from .judge import TARGET, VerificationFirstJudge
from .memory import Memory
from .researcher import PolynomialResearcher
from .tools import ToolRunner
from .tracking import canonical, digest, write_json
from .verifiers import MeasurementVerifier, ProvenanceVerifier, claim_identity


def validate_case(case):
    required = {"id", "question", "topic", "target_x", "sources", "experiment"}
    if not required <= set(case) or not all(isinstance(case[k], str) and case[k].strip() for k in ("id", "question", "topic")):
        raise ContractError("Case requires id, question, topic, target_x, sources and experiment")
    finite_number(case["target_x"], 1e4)
    if not isinstance(case["sources"], list) or len(case["sources"]) > 20:
        raise ContractError("Expected at most 20 local sources")
    for src in case["sources"]:
        if not {"title", "uri", "version", "rights", "text"} <= set(src):
            raise ContractError("Source metadata is incomplete")
        if not isinstance(src["text"], str) or len(src["text"]) > 20000:
            raise ContractError("Source text exceeds miniature limits")
    ex = case["experiment"]
    if set(ex) != {"available", "observations"} or type(ex["available"]) is not bool or len(ex["observations"]) > 100:
        raise ContractError("Invalid experiment contract")
    for x, y in ex["observations"]:
        finite_number(x, 1e4)
        finite_number(y)


class Controller:
    def __init__(self, researcher=None, judge=None, tools=None, max_actions=8, timeout=5.0):
        if type(max_actions) is not int or not 0 <= max_actions <= 100:
            raise ContractError("max_actions must be integer in [0,100]")
        self.researcher = researcher or PolynomialResearcher()
        self.judge = judge or VerificationFirstJudge()
        self.tools = tools or ToolRunner()
        self.max_actions = max_actions
        self.timeout = timeout

    def run(self, case, run):
        validate_case(case)
        state = ResearchState("1", case["question"], case["topic"], case["target_x"])
        state.assumptions = ["Synthetic numerical domain; demonstration of architecture only",
                             "Verification is local agreement at one measurement, not scientific law discovery",
                             "Researcher backend: " + self.researcher.model_id]
        memory = Memory(run.path / "memory")
        used = 0

        def snapshot(phase):
            state.phase = phase
            memory.append("STATE", state.to_dict())
            run.event("PHASE", {"phase": phase})

        def invoke(name, args):
            nonlocal used
            if used >= self.max_actions:
                raise ContractError("Action budget exhausted")
            used += 1
            action = Action(f"action-{used}", name, args, self.timeout)
            memory.append("ACTION", asdict(action))
            result = self.tools.execute(action)
            if result.action_id != action.id:
                raise ContractError("Tool result is bound to a different action")
            if result.outcome == Outcome.PASS:
                finite_number(result.value)
            state.results.append(result)
            memory.append("RESULT", asdict(result))
            return result

        try:
            snapshot("CREATED")
            state.plan = self.researcher.plan(copy.deepcopy(state))
            snapshot("PLANNED")
            for src in case["sources"]:
                memory.ingest(**src)
            state.evidence = memory.search(case["topic"])
            for ev in state.evidence:
                # Only structured observation records are interpreted. Other source prose is inert.
                try:
                    record = json.loads(ev.quote)
                except ValueError:
                    continue
                if not isinstance(record, dict) or record.get("topic") != case["topic"]:
                    continue
                rows = record.get("observations", [])
                if not isinstance(rows, list) or len(rows) > 100:
                    raise ContractError("Invalid observation list")
                for pair in rows:
                    if not isinstance(pair, list) or len(pair) != 2:
                        raise ContractError("Observations require [x,y]")
                    state.observations.append([finite_number(pair[0], 1e4), finite_number(pair[1])])
            snapshot("RETRIEVED")
            unique = {}
            for x, y in state.observations:
                if x in unique and y != unique[x]:
                    state.contradictions.append({"x": x, "values": [unique[x], y], "status": "unresolved source conflict"})
                unique[x] = y
            state.observations = [[x, y] for x, y in sorted(unique.items())]
            if state.contradictions:
                state.unknowns.append("Conflicting observations require source adjudication")
            elif len(state.observations) < 2:
                state.unknowns.append("Insufficient relevant observations to propose supported candidates")
            else:
                try:
                    state.hypotheses = self.researcher.hypothesize(copy.deepcopy(state))
                finally:
                    trace = getattr(self.researcher, "last_trace", None)
                    if trace is not None:
                        memory.append("RESEARCHER_OUTPUT", trace)
                        run.event("RESEARCHER_OUTPUT", trace)
                ids = [h.id for h in state.hypotheses]
                if len(ids) != len(set(ids)) or len(ids) > 3:
                    raise ContractError("Duplicate or excessive candidate hypotheses")
                valid_evidence = {e.id for e in state.evidence}
                if any(not set(h.evidence_ids) <= valid_evidence for h in state.hypotheses):
                    raise ContractError("Researcher fabricated evidence references")
                snapshot("HYPOTHESIZED")
                for h in state.hypotheses:
                    calculated = invoke("CALCULATE", {"coefficients": h.coefficients, "x": state.target_x})
                    if calculated.outcome != Outcome.PASS:
                        state.unknowns.append(f"Calculation for {h.id}: {calculated.outcome.value}")
                        continue
                    claim = Claim("c-" + h.id, h.id, f"y({state.target_x}) = {calculated.value}",
                                  calculated.value, state.target_x, list(h.evidence_ids))
                    claim.forecast = self.judge.decide(copy.deepcopy(state), h, copy.deepcopy(claim))
                    if claim.forecast.claim_id != claim.id or claim.forecast.target != TARGET:
                        raise ContractError("Judge forecast is bound to a different claim or event")
                    state.claims.append(claim)
                    memory.append("FORECAST_BEFORE_MEASUREMENT", asdict(claim.forecast))
                    for eid in claim.evidence_ids:
                        memory.edge(claim.id, "DERIVED_FROM", eid)
                snapshot("PREDICTED")
                requested = [c for c in state.claims if c.forecast.action == "VERIFY"]
                if requested:
                    # Only the controller passes the experimental observations to the measurement worker.
                    measured = invoke("MEASURE", {"x": state.target_x, **case["experiment"]})
                    measured_evidence = None
                    if measured.outcome == Outcome.PASS:
                        text = canonical({"x": state.target_x, "value": measured.value, "result": asdict(measured)})
                        sid = memory.ingest(text, uri=f"aim://experiment/{case['id']}", version="1",
                                            rights="project-generated-fixture", title="Independent synthetic measurement", retrievable=False)
                        measured_evidence = memory.span(sid, 0, len(text))
                        state.evidence.append(measured_evidence)
                    for claim in requested:
                        provenance = ProvenanceVerifier().verify_spans(claim, memory, state.evidence)
                        numeric = MeasurementVerifier().verify(claim, measured, measured_evidence)
                        state.verifications.extend([provenance, numeric])
                        claim.verification_ids = [provenance.id, numeric.id]
                        for v in (provenance, numeric):
                            memory.append("VERIFICATION", asdict(v))
                            memory.edge(claim.id, "CHECKED_BY", v.id)
                            for eid in v.evidence_ids:
                                memory.edge(v.id, "USED", eid)
                        before = claim.status
                        claim.status = (Status.VERIFIED if numeric.outcome == Outcome.PASS else
                                        Status.CONTRADICTED if numeric.outcome == Outcome.FAIL else Status.UNKNOWN)
                        if provenance.outcome != Outcome.PASS:
                            claim.status = Status.UNKNOWN
                        memory.append("CLAIM_REVISED", {"claim_id": claim.id, "from": before, "to": claim.status,
                                                        "verification_ids": claim.verification_ids})
                        if claim.status == Status.CONTRADICTED:
                            state.contradictions.append({"claim_id": claim.id, "verification_id": numeric.id,
                                                         "status": "prediction rejected; history retained"})
                for c in state.claims:
                    if c.status == Status.HYPOTHESIS:
                        c.status = Status.UNKNOWN
                        state.unknowns.append(f"Judge abstained on {c.id}")
                snapshot("JUDGED_AND_REVISED")
        except ContractError as exc:
            state.unknowns.append(str(exc))
            memory.append("CONTROLLED_FAILURE", {"error": str(exc)})
            for claim in state.claims:
                if claim.status == Status.HYPOTHESIS:
                    claim.status = Status.UNKNOWN
        except BaseException as exc:
            memory.append("FAILURE", {"type": type(exc).__name__, "message": str(exc)})
            memory.close()
            raise
        try:
            self._validate_final(state, memory)
            state.final_response = render(state)
            snapshot("FINAL")
            if canonical(memory.replay()) != canonical(state.to_dict()):
                raise ContractError("State replay diverged")
            write_json(run.path / "state.json", state.to_dict())
            (run.path / "response.md").write_text(state.final_response, encoding="utf-8")
            write_json(run.path / "loop-summary.json", {"action_count": used, "researcher": self.researcher.model_id,
                       "judge": self.judge.model_id, "verified_claims": sum(c.status == Status.VERIFIED for c in state.claims),
                       "unresolved": state.unknowns, "scope": "synthetic engineering fixture"})
            return state
        finally:
            memory.close()

    @staticmethod
    def _validate_final(state, memory):
        for ev in state.evidence:
            if not memory.validate(ev):
                raise ContractError("Final evidence integrity check failed")
        index = {v.id: v for v in state.verifications}
        for c in state.claims:
            if c.status == Status.VERIFIED:
                checks = [index[i] for i in c.verification_ids]
                if ({v.verifier for v in checks} != {"numeric-observation-agreement", "source-span-integrity"}
                    or any(v.outcome != Outcome.PASS or v.claim_hash != digest(claim_identity(c)) for v in checks)):
                    raise ContractError("Verified claim lacks passing checks bound to its contents")


def render(state):
    lines = ["# AIM Model-1 investigation", "", state.question, "", "## Findings", ""]
    if not state.claims:
        lines.append("No supported candidate could be evaluated. The question remains unresolved.")
    for c in state.claims:
        lines += [f"- {c.id}: {c.text} — {c.status.value}. Evidence: {', '.join(c.evidence_ids)}."]
        if c.forecast.probability is not None:
            lines.append(f"  Pre-measurement forecast: {c.forecast.probability:.4f}; target: {c.forecast.target}.")
        for v in state.verifications:
            if v.claim_id == c.id:
                lines.append(f"  Check {v.id}: {v.verifier}@{v.version} = {v.outcome.value}. Scope: {v.scope}. Evidence: {', '.join(v.evidence_ids)}.")
    lines += ["", "## Unresolved and limitations", ""] + ["- " + x for x in state.unknowns + state.assumptions]
    lines += ["", "## Evidence", ""]
    for e in state.evidence:
        lines.append(f"- {e.id}: [{e.source_id}](memory/objects/{e.source_hash}), {e.locator}, SHA-256 {e.source_hash}.")
    return "\n".join(lines) + "\n"

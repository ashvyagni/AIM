"""Separate symbolic contracts and an end-to-end provenance-backed research loop."""
from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from .contracts import Action, ContractError, Decision, Evidence, Outcome, Status, Verification
from .memory import Memory
from .symbolic import ASSUMPTIONS, SCHEMA, SCOPE, VERSION, check_identity, identity_request
from .tools import ToolRunner
from .tracking import canonical, digest, write_json

TARGET = "exact-polynomial-checker-v1 returns PASS for this candidate under Q[x] bounds"
RESEARCH_CONTRACT = "aim-symbolic-researcher-v1"


@dataclass(frozen=True)
class SymbolicHypothesis:
    id: str
    rhs: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id or not isinstance(self.rhs, str) or not 1 <= len(self.rhs) <= 512:
            raise ContractError("symbolic hypothesis needs an ID and bounded expression")
        if not self.evidence_ids or len(self.evidence_ids) > 5 or any(not isinstance(i, str) for i in self.evidence_ids) or len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ContractError("symbolic hypothesis needs distinct source references")


@dataclass
class SymbolicState:
    question: str
    lhs: str
    schema: str = SCHEMA
    assumptions: dict = field(default_factory=lambda: dict(ASSUMPTIONS))
    phase: str = "CREATED"
    plan: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    hypotheses: list = field(default_factory=list)
    claims: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    results: list = field(default_factory=list)
    verifications: list = field(default_factory=list)
    unknowns: list = field(default_factory=list)
    final_response: str = ""

    def to_dict(self):
        return asdict(self)


class SymbolicResearcher(Protocol):
    model_id: str
    def plan(self, state: SymbolicState) -> list[str]: ...
    def hypothesize(self, state: SymbolicState) -> list[SymbolicHypothesis]: ...


class SymbolicJudge(Protocol):
    model_id: str
    def decide(self, state: SymbolicState, claim: dict) -> Decision: ...


class BinomialResearcher:
    """Independent elementary formula baseline, only (x +/- integer)**2.

    This is a deterministic proposer, not a trained language model. It does not
    import or call the polynomial normalizer to manufacture a verified answer.
    """
    model_id = "binomial-formula-reference-v1"

    def plan(self, state):
        return ["Retrieve the declared expression and assumptions", "Propose a bounded expansion",
                "Ask the separate Judge whether to check", "Check exact identity and cited spans", "Report the scoped result"]

    def hypothesize(self, state):
        match = re.fullmatch(r"\(x([+-])(\d{1,3})\)\*\*2", state.lhs)
        if not match:
            return []
        a = int(match[2]) * (-1 if match[1] == "-" else 1)
        return [SymbolicHypothesis("h1", f"x**2+({2*a})*x+({a*a})", (state.evidence[0].id,))]


class SymbolicVerificationFirstJudge:
    model_id = "symbolic-verification-first-v1"

    def decide(self, state, claim):
        return Decision(claim["id"], None, TARGET, "VERIFY", self.model_id,
                        "No learned symbolic calibration; numeric-domain forecasts are inapplicable")


def symbolic_prompt(lhs):
    # The matched evidence alias is constant, but maps to an immutable span per run.
    return f'Expand in Q[x]. Source E0 expression: {lhs}\nReturn JSON rhs and evidence_ids.\n'


def parse_symbolic_response(text, evidence):
    if not isinstance(text, str) or len(text) > 2048:
        raise ContractError("symbolic response too large")
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ContractError("duplicate symbolic response key")
            result[k] = v
        return result
    try:
        value = json.loads(text, object_pairs_hook=unique)
    except (ValueError, RecursionError) as exc:
        raise ContractError("invalid symbolic response JSON") from exc
    if not isinstance(value, dict) or set(value) != {"rhs", "evidence_ids"} or value["evidence_ids"] != ["E0"] or not evidence:
        raise ContractError("symbolic output requires rhs and the available E0 source")
    return SymbolicHypothesis("h1", value["rhs"], (evidence[0].id,))


class SymbolicTransformerResearcher(BinomialResearcher):
    def __init__(self, checkpoint, max_new_tokens=96):
        from .neural import load_lm
        from .tracking import file_hash
        self.model, self.tokenizer, record = load_lm(checkpoint)
        if record.get("research_contract") != RESEARCH_CONTRACT:
            raise ContractError("checkpoint is not a symbolic Researcher")
        self.model.eval()
        self.model_id = "native-symbolic-" + file_hash(checkpoint)[:16]
        self.max_new_tokens = max_new_tokens
        self.last_trace = None

    def hypothesize(self, state):
        prompt = symbolic_prompt(state.lhs)
        raw = self.model.generate_text(self.tokenizer, prompt, self.max_new_tokens)
        self.last_trace = {"prompt": prompt, "raw": raw, "model_id": self.model_id}
        return [parse_symbolic_response(raw, state.evidence)]


def claim_identity(claim):
    return {k: claim[k] for k in ("id", "hypothesis_id", "request", "evidence_ids")}


class SymbolicVerifier:
    name = "exact-polynomial-identity"

    def verify(self, claim, result):
        identity = claim_identity(claim)
        if result.action_id != "check-" + claim["id"]:
            raise ContractError("tool result action binding mismatch")
        outcome, detail = result.outcome, result.detail
        if outcome == Outcome.PASS:
            # Recompute the bounded certificate. A forged/misbound worker PASS
            # cannot promote a claim; this adds CPU work, not another tool call.
            expected = check_identity(claim["request"])
            if canonical(result.value) != canonical(expected):
                raise ContractError("symbolic certificate mismatch")
            outcome, detail = Outcome(expected["outcome"]), expected["detail"]
        elif outcome == Outcome.FAIL:
            raise ContractError("transport failure cannot assert a false identity")
        return Verification("vs-" + digest(identity)[:24], claim["id"], digest(identity), self.name,
                            VERSION, outcome, SCOPE, tuple(claim["evidence_ids"]), detail)


def validate_case(case):
    required = {"schema", "id", "question", "lhs", "assumptions", "sources"}
    if not isinstance(case, dict) or set(case) != required or case["schema"] != SCHEMA or case["assumptions"] != ASSUMPTIONS:
        raise ContractError("invalid symbolic case contract or assumptions")
    for key, limit in (("id", 128), ("question", 2048), ("lhs", 512)):
        if not isinstance(case[key], str) or not 1 <= len(case[key]) <= limit:
            raise ContractError("invalid symbolic case text")
    if not isinstance(case["sources"], list) or not 1 <= len(case["sources"]) <= 5:
        raise ContractError("symbolic case needs 1..5 sources")
    for source in case["sources"]:
        if not isinstance(source, dict) or set(source) != {"text", "title", "uri", "version", "rights"}:
            raise ContractError("invalid symbolic source contract")
        if any(not isinstance(v, str) or not 1 <= len(v) <= 4096 for v in source.values()):
            raise ContractError("invalid symbolic source text")


class SymbolicController:
    def __init__(self, researcher=None, judge=None, tools=None, max_actions=3, timeout=5):
        if type(max_actions) is not int or not 0 <= max_actions <= 3:
            raise ContractError("symbolic action budget must be 0..3")
        Action("validate", "CHECK_POLYNOMIAL_IDENTITY", {}, timeout)
        self.researcher = researcher or BinomialResearcher()
        self.judge = judge or SymbolicVerificationFirstJudge()
        self.tools = tools or ToolRunner()
        self.max_actions, self.timeout = max_actions, timeout

    def run(self, case, run):
        validate_case(case)
        state = SymbolicState(case["question"], case["lhs"])
        memory = Memory(run.path/"memory")

        def snapshot(phase):
            state.phase = phase
            memory.append("STATE", state.to_dict())
            run.event("PHASE", {"phase": phase})

        try:
            snapshot("CREATED")
            try:
                state.plan = self.researcher.plan(copy.deepcopy(state))
                if not isinstance(state.plan, list) or not 1 <= len(state.plan) <= 10 or any(not isinstance(x, str) or len(x) > 1024 for x in state.plan):
                    raise ContractError("invalid symbolic plan")
                snapshot("PLANNED")
                for source in case["sources"]:
                    memory.ingest(**source)
                # Retrieval supplies evidence, never an instruction to execute.
                for ev in memory.search("expression", limit=5):
                    try:
                        record = json.loads(ev.quote)
                    except ValueError:
                        continue
                    if isinstance(record, dict) and record == {"expression": state.lhs, "assumptions": ASSUMPTIONS}:
                        state.evidence.append(ev)
                snapshot("RETRIEVED")
                if not state.evidence:
                    raise ContractError("no exact source declaration matches the question and assumptions")
                try:
                    proposals = self.researcher.hypothesize(copy.deepcopy(state))
                finally:
                    if getattr(self.researcher, "last_trace", None) is not None:
                        write_json(run.path/"researcher-output.json", self.researcher.last_trace)
                if not isinstance(proposals, list) or len(proposals) > 3 or any(not isinstance(h, SymbolicHypothesis) for h in proposals) or len({h.id for h in proposals}) != len(proposals):
                    raise ContractError("invalid symbolic hypothesis collection")
                if not proposals:
                    state.unknowns.append("Researcher proposed no supported candidate; no fallback")
                index = {e.id: e for e in state.evidence}
                if any(i not in index for h in proposals for i in h.evidence_ids):
                    raise ContractError("hypothesis cites unavailable evidence")
                state.hypotheses = proposals
                snapshot("PROPOSED")
                for h in proposals:
                    claim = {"id": "c-"+h.id, "hypothesis_id": h.id, "request": identity_request(state.lhs, h.rhs),
                             "evidence_ids": list(h.evidence_ids), "status": Status.UNKNOWN.value}
                    state.claims.append(claim)
                    decision = self.judge.decide(copy.deepcopy(state), copy.deepcopy(claim))
                    if not isinstance(decision, Decision) or decision.claim_id != claim["id"] or decision.target != TARGET:
                        raise ContractError("symbolic Judge event/claim mismatch")
                    state.decisions.append(decision)
                    memory.append("DECISION", asdict(decision))
                    snapshot("JUDGED")
                    if decision.action == "ABSTAIN" or len(state.results) >= self.max_actions:
                        state.unknowns.append(claim["id"]+": abstention or exhausted action budget")
                        continue
                    action = Action("check-"+claim["id"], "CHECK_POLYNOMIAL_IDENTITY", claim["request"], self.timeout)
                    memory.append("DISPATCH", asdict(action))
                    result = self.tools.execute(action)
                    state.results.append(result)
                    memory.append("TOOL_RESULT", asdict(result))
                    verification = SymbolicVerifier().verify(claim, result)
                    good = all(memory.validate(index[i]) for i in claim["evidence_ids"])
                    provenance = Verification("vp-"+digest(claim_identity(claim))[:24], claim["id"], digest(claim_identity(claim)),
                                              "source-span-integrity", "1", Outcome.PASS if good else Outcome.FAIL,
                                              "Source/version/span identity only; no natural-language entailment",
                                              tuple(claim["evidence_ids"]), "cited spans checked against originals")
                    state.verifications.extend([verification, provenance])
                    for v in (verification, provenance):
                        memory.append("VERIFICATION", asdict(v))
                        memory.edge(claim["id"], "CHECKED_BY", v.id)
                    for eid in claim["evidence_ids"]:
                        memory.edge(claim["id"], "CITES", eid)
                    if good and verification.outcome in {Outcome.PASS, Outcome.FAIL}:
                        claim["status"] = Status.VERIFIED.value if verification.outcome == Outcome.PASS else Status.CONTRADICTED.value
                    else:
                        state.unknowns.append(claim["id"]+": "+verification.detail)
                    snapshot("REVISED")
            except ContractError as exc:
                state.unknowns.append(str(exc))
                memory.append("CONTROLLED_FAILURE", {"error": str(exc)})
            _validate_state(state.to_dict(), memory)
            state.final_response = render(state)
            snapshot("COMPLETE")
            write_json(run.path/"state.json", state.to_dict())
            (run.path/"response.md").write_text(state.final_response, encoding="utf-8")
            if canonical(memory.replay()) != canonical(state.to_dict()):
                raise ContractError("symbolic ledger replay differs from final state")
            return state
        finally:
            memory.close()


def _validate_state(state, memory):
    if state["schema"] != SCHEMA or state["assumptions"] != ASSUMPTIONS:
        raise ContractError("symbolic state assumptions changed")
    index = {e["id"]: Evidence(**e) for e in state["evidence"]}
    if not all(memory.validate(e) for e in index.values()):
        raise ContractError("symbolic evidence integrity failure")
    for claim in state["claims"]:
        if claim["request"]["lhs"] != state["lhs"] or not claim["evidence_ids"] or any(i not in index for i in claim["evidence_ids"]):
            raise ContractError("symbolic claim/source binding failure")
        if claim["status"] in {Status.VERIFIED.value, Status.CONTRADICTED.value}:
            expected = "PASS" if claim["status"] == Status.VERIFIED.value else "FAIL"
            checks = [v for v in state["verifications"] if v["claim_id"] == claim["id"]]
            if len(checks) != 2 or {v["verifier"] for v in checks} != {SymbolicVerifier.name, "source-span-integrity"}:
                raise ContractError("resolved symbolic claim lacks independent checks")
            for v in checks:
                want = expected if v["verifier"] == SymbolicVerifier.name else "PASS"
                if v["outcome"] != want or v["claim_hash"] != digest(claim_identity(claim)) or list(v["evidence_ids"]) != claim["evidence_ids"]:
                    raise ContractError("symbolic check binding mismatch")
            if check_identity(claim["request"])["outcome"] != expected:
                raise ContractError("symbolic final check failed")


def render(state):
    lines = ["# Symbolic research response", "", state.question, "", "Scope: "+SCOPE+".", "",
             "Assumptions: x is an indeterminate over the rationals. No variable denominators.", ""]
    for claim in state.claims:
        lines.append(f'- **{claim["status"]}** `{claim["request"]["lhs"]} = {claim["request"]["rhs"]}` ({claim["id"]}).')
        for v in state.verifications:
            if v.claim_id == claim["id"]:
                lines.append(f"  - Check {v.id}: {v.verifier} — {v.outcome.value}: {v.detail}.")
    lines += ["", "## Sources", ""]
    lines += [f"- [{e.id}](memory/objects/{e.source_hash}) ({e.locator}; SHA-256 {e.source_hash})." for e in state.evidence]
    if state.unknowns:
        lines += ["", "## Unresolved", ""] + ["- "+u for u in state.unknowns]
    return "\n".join(lines)+"\n"


def audit_symbolic_run(path):
    path = Path(path)
    state = json.loads((path/"state.json").read_text())
    memory = Memory(path/"memory", read_only=True)
    try:
        if canonical(memory.replay()) != canonical(state):
            raise ContractError("symbolic audit replay mismatch")
        _validate_state(state, memory)
        # Decisions must precede actual dispatches for the same claim.
        seen, dispatched = {}, set()
        for (record,) in memory.db.execute("SELECT record FROM events ORDER BY seq"):
            event = json.loads(record)
            payload = event["payload"]
            if event["kind"] == "DECISION":
                if payload["claim_id"] in seen:
                    raise ContractError("duplicate symbolic decision")
                seen[payload["claim_id"]] = payload
            if event["kind"] == "DISPATCH":
                cid = payload["id"].removeprefix("check-")
                if cid in dispatched or cid not in seen or seen[cid]["action"] != "VERIFY" or seen[cid]["target"] != TARGET:
                    raise ContractError("symbolic dispatch lacks a prior valid decision")
                dispatched.add(cid)
        if len(dispatched) != len(state["results"]):
            raise ContractError("symbolic dispatch/result accounting mismatch")
        return {"claims": len(state["claims"]), "verified": sum(c["status"] == "VERIFIED" for c in state["claims"]),
                "dispatches": len(dispatched), "replay_and_checks": "PASS"}
    finally:
        memory.close()

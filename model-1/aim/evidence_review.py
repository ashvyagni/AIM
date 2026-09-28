"""Bounded document-review loop with separate proposal, decision and verification.

The reference Researcher is extractive. It does not pretend to be a trained
scientific model. Citations can pass provenance checks; their assertions remain
UNVERIFIED until a domain verifier exists.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from .contracts import ContractError, Evidence
from .corpus import read_json, require
from .library import Library, bounded_text
from .memory import Memory
from .tracking import Run, digest, write_json

SCHEMA = "aim-evidence-review-v1"


@dataclass(frozen=True)
class ReviewPlan:
    queries: tuple[str, ...]
    limit: int = 8
    per_source: int = 2


@dataclass(frozen=True)
class ExcerptProposal:
    chunk_id: str
    text: str


class DocumentResearcher(Protocol):
    model_id: str

    def plan(self, question: str) -> ReviewPlan: ...

    def propose(self, question: str, hits: list[dict]) -> list[ExcerptProposal]: ...


class DocumentJudge(Protocol):
    model_id: str

    def decide(self, proposal: ExcerptProposal, verification: dict) -> dict: ...


class ExtractiveResearcher:
    model_id = "reference-extractive-document-researcher-v1"

    def plan(self, question):
        return ReviewPlan((question,))

    def propose(self, question, hits):
        return [ExcerptProposal(h["chunk_id"], h["quote"]) for h in hits]


class ProvenanceJudge:
    model_id = "reference-provenance-judge-v1"

    def decide(self, proposal, verification):
        return {"action": "CITE" if verification["outcome"] == "PASS" else "ABSTAIN",
                "probability": None, "target": "exact-source-attribution",
                "calibration_status": "NOT_CALIBRATED", "model_id": self.model_id}


def validate_decision(decision, model_id=None):
    require(isinstance(decision, dict) and set(decision) == {"action", "probability", "target", "calibration_status", "model_id"}
            and decision["action"] in {"CITE", "ABSTAIN"} and decision["probability"] is None
            and decision["target"] == "exact-source-attribution" and decision["calibration_status"] == "NOT_CALIBRATED",
            "invalid document judge decision")
    bounded_text(decision["model_id"], "document judge ID")
    require(model_id is None or decision["model_id"] == model_id, "document judge identity mismatch")


class ExcerptVerifier:
    name = "exact-source-attribution"
    version = "1"

    def verify(self, proposal, hit, evidence, memory):
        passed = (proposal.chunk_id == hit["chunk_id"] and proposal.text == hit["quote"]
                  and evidence.quote == hit["quote"] and evidence.quote_hash == hit["quote_hash"]
                  and evidence.source_hash == hit["metadata"]["sha256"] and evidence.start == hit["start"]
                  and evidence.end == hit["end"] and memory.validate(evidence))
        return {"verifier": self.name, "version": self.version, "outcome": "PASS" if passed else "FAIL",
                "proposal_hash": digest(asdict(proposal)), "evidence_id": evidence.id,
                "scope": "exact excerpt and source bytes; no entailment, correctness or independence proof"}


@dataclass
class ReviewState:
    question: str
    schema: str = SCHEMA
    phase: str = "CREATED"
    plan: dict = field(default_factory=dict)
    snapshot: dict = field(default_factory=dict)
    packet_hashes: list[str] = field(default_factory=list)
    proposals_hash: str = ""
    excerpts: list[dict] = field(default_factory=list)
    abstentions: list[dict] = field(default_factory=list)
    limitations: list[str] = field(default_factory=lambda: [
        "Source statements are UNVERIFIED. Exact attribution does not establish factual truth.",
        "Lexical retrieval is incomplete and may miss paraphrases or split terms at chunk boundaries.",
        "Versions and mirrors of a source are not independent corroboration.",
        "The reference Researcher extracts text; no learned synthesis or hypothesis testing is performed.",
        "The reference Judge has no learned or calibrated probability.",
        "Sources are active at the recorded snapshot; consult the current library for later retirement.",
    ])
    final_response: str = ""


def fence(text):
    # Source text can contain Markdown, HTML, instructions or fence delimiters.
    marker = "`" * (max([len(part) for part in re.findall(r"`+", text)] + [2]) + 1)
    return marker + "text\n" + text + "\n" + marker


def render(state):
    lines = ["# AIM local evidence review", "", "## Question", "", fence(state.question), "",
             "Source excerpts below are **UNVERIFIED assertions**. A PASS applies only to attribution.", ""]
    if not state.excerpts:
        lines += ["No excerpts passed the selection and attribution checks. No factual answer is asserted.", ""]
    for index, row in enumerate(state.excerpts, 1):
        lines += [f"## Excerpt {index} — UNVERIFIED", "", fence(row["text"]), "",
                  fence(json.dumps({k: row[k] for k in ("uri", "version", "title", "source_id", "chunk_id", "start", "end", "evidence_id")}, ensure_ascii=False, indent=2)), "",
                  "Attribution check: PASS. Character offsets refer to the unchanged original.", ""]
    lines += ["## Limits", ""] + ["- " + item for item in state.limitations]
    lines += ["", "## Snapshot", "", fence(json.dumps(state.snapshot, sort_keys=True)), ""]
    return "\n".join(lines)


class EvidenceReviewController:
    def __init__(self, researcher=None, judge=None):
        self.researcher = researcher or ExtractiveResearcher()
        self.judge = judge or ProvenanceJudge()
        self.verifier = ExcerptVerifier()

    def run(self, question, library_path, runs):
        bounded_text(question, "research question", 2048)
        config = {"schema": SCHEMA, "question": question, "researcher": self.researcher.model_id,
                  "judge": self.judge.model_id, "max_queries": 3, "max_excerpts": 32,
                  "library_path": str(Path(library_path).resolve())}
        with Run(Path(runs), "evidence-review", config) as run:
            with Library(library_path, read_only=True) as live:
                audit = live.backup(run.path / "library.sqlite")
            write_json(run.path / "library-audit.json", audit)
            state = ReviewState(question, snapshot=audit["snapshot"])
            memory = Memory(run.path / "memory")
            try:
                def record(phase):
                    state.phase = phase
                    memory.append("STATE", asdict(state))
                    run.event("PHASE", {"phase": phase})

                record("CREATED")
                plan = self.researcher.plan(question)
                require(isinstance(plan, ReviewPlan) and isinstance(plan.queries, tuple) and 1 <= len(plan.queries) <= 3
                        and len(set(plan.queries)) == len(plan.queries), "invalid document research plan")
                for query in plan.queries:
                    bounded_text(query, "planned query", 2048)
                require(type(plan.limit) is int and 1 <= plan.limit <= 32 and type(plan.per_source) is int
                        and 1 <= plan.per_source <= 8, "invalid document retrieval budget")
                state.plan = asdict(plan)
                # Persisted JSON must have the same form as ledger replay.
                state.plan["queries"] = list(plan.queries)
                record("PLANNED")
                packets, hits = [], {}
                with Library(run.path / "library.sqlite", read_only=True) as library:
                    for query in plan.queries:
                        packet = library.search(query, limit=plan.limit, per_source=plan.per_source)
                        library.verify_packet(packet)
                        packets.append(packet)
                        state.packet_hashes.append(packet["packet_hash"])
                        for hit in packet["hits"]:
                            hits.setdefault(hit["chunk_id"], hit)
                    write_json(run.path / "retrieval.json", packets)
                    record("RETRIEVED")
                    bounded_hits = list(hits.values())[:32]
                    proposals = self.researcher.propose(question, copy.deepcopy(bounded_hits))
                    require(isinstance(proposals, list) and len(proposals) <= 32
                            and all(isinstance(p, ExcerptProposal) for p in proposals)
                            and len({p.chunk_id for p in proposals}) == len(proposals), "invalid excerpt proposals")
                    allowed = {h["chunk_id"] for h in bounded_hits}
                    write_json(run.path / "proposals.json", [asdict(p) for p in proposals])
                    state.proposals_hash = digest([asdict(p) for p in proposals])
                    record("PROPOSED")
                    for proposal in proposals:
                        require(proposal.chunk_id in allowed, "Researcher proposed an unretrieved chunk")
                        bounded_text(proposal.text, "excerpt proposal", 8192)
                        hit = hits[proposal.chunk_id]
                        source = library.source(hit["source_id"])
                        meta = source["metadata"]
                        sid = memory.ingest(source["text"], uri=meta["uri"], version=meta["version"],
                                            rights=meta["rights"], title=meta["title"])
                        evidence = memory.span(sid, hit["start"], hit["end"])
                        check = self.verifier.verify(proposal, hit, evidence, memory)
                        decision = self.judge.decide(copy.deepcopy(proposal), copy.deepcopy(check))
                        validate_decision(decision, self.judge.model_id)
                        memory.append("ATTRIBUTION_CHECK", check)
                        memory.append("JUDGEMENT", decision)
                        if check["outcome"] == "PASS" and decision["action"] == "CITE":
                            row = {"chunk_id": proposal.chunk_id, "source_id": hit["source_id"], "memory_source_id": sid,
                                   "text": proposal.text, "status": "UNVERIFIED", "evidence_id": evidence.id,
                                   "start": evidence.start, "end": evidence.end, "uri": meta["uri"],
                                   "title": meta["title"], "version": meta["version"], "verification": check, "decision": decision}
                            state.excerpts.append(row)
                            memory.edge(evidence.id, "IMPORTED_FROM", hit["chunk_id"])
                        else:
                            state.abstentions.append({"proposal": asdict(proposal), "verification": check, "decision": decision})
                        record("REVISED")
                state.final_response = render(state)
                record("FINAL")
                require(memory.replay() == asdict(state), "document review ledger replay mismatch")
                write_json(run.path / "state.json", asdict(state))
                (run.path / "response.md").write_text(state.final_response, encoding="utf-8")
                write_json(run.path / "summary.json", {"queries": len(packets), "retrieved_chunks": len(hits),
                           "selected_excerpts": len(state.excerpts), "abstentions": len(state.abstentions),
                           "verified_factual_claims": 0, "snapshot": state.snapshot})
            finally:
                memory.close()
            write_json(run.path / "replay-audit.json", audit_review(run.path))
        return run.path


def audit_review(path, current_library=None):
    path = Path(path)
    state = read_json(path / "state.json")
    require(state.get("schema") == SCHEMA and state.get("phase") == "FINAL", "incomplete document review")
    packets = read_json(path / "retrieval.json")
    proposals = read_json(path / "proposals.json")
    require(isinstance(proposals, list) and len(proposals) <= 32 and digest(proposals) == state["proposals_hash"]
            and all(isinstance(p, dict) and set(p) == {"chunk_id", "text"} for p in proposals)
            and len({p["chunk_id"] for p in proposals}) == len(proposals), "review proposal hash or contract mismatch")
    require(isinstance(packets, list) and 1 <= len(packets) <= 3
            and state["packet_hashes"] == [p["packet_hash"] for p in packets]
            and state["plan"]["queries"] == [p["query"] for p in packets], "review retrieval binding mismatch")
    hit_map = {}
    with Library(path / "library.sqlite", read_only=True) as library:
        audit = library.audit()
        require(audit["snapshot"] == state["snapshot"], "review library snapshot mismatch")
        for packet in packets:
            require(packet["snapshot"] == state["snapshot"] and packet["limit"] == state["plan"]["limit"]
                    and packet["per_source"] == state["plan"]["per_source"], "review query budget mismatch")
            library.verify_packet(packet)
            for hit in packet["hits"]:
                hit_map.setdefault(hit["chunk_id"], hit)
    memory = Memory(path / "memory", read_only=True)
    try:
        require(memory.replay() == state, "review state ledger mismatch")
        allowed = set(list(hit_map)[:32])
        require(all(p["chunk_id"] in allowed for p in proposals), "review proposal escaped retrieval budget")
        manifest = read_json(path / "manifest.json", limit=16*1024*1024)
        judge_id = manifest["configuration"]["judge"]
        def recheck(proposal, recorded):
            hit = hit_map[proposal.chunk_id]
            record = memory.db.execute("SELECT record FROM evidence WHERE id=?", (recorded["evidence_id"],)).fetchone()
            require(record is not None, "review citation missing from memory")
            ev = Evidence(**json.loads(record[0]))
            _, meta = memory.source(ev.source_id)
            require(all(meta[k] == hit["metadata"][k] for k in ("uri", "version", "title", "rights", "sha256")), "review source metadata mismatch")
            check = ExcerptVerifier().verify(proposal, hit, ev, memory)
            require(check == recorded, "review verifier result does not replay")
            return ev, check

        expected_proposals = []
        for row in state["excerpts"]:
            hit = hit_map[row["chunk_id"]]
            require(row["status"] == "UNVERIFIED" and row["source_id"] == hit["source_id"]
                    and row["text"] == hit["quote"] and row["start"] == hit["start"] and row["end"] == hit["end"]
                    and all(row[k] == hit["metadata"][k] for k in ("uri", "title", "version")), "review excerpt binding mismatch")
            proposal = ExcerptProposal(row["chunk_id"], row["text"])
            ev, check = recheck(proposal, row["verification"])
            validate_decision(row["decision"], judge_id)
            require(ev.source_id == row["memory_source_id"] and ev.id == row["evidence_id"] and check["outcome"] == "PASS"
                    and row["decision"]["action"] == "CITE", "review attribution did not pass")
            expected_proposals.append(asdict(proposal))
        for row in state["abstentions"]:
            proposal = ExcerptProposal(**row["proposal"])
            _, check = recheck(proposal, row["verification"])
            validate_decision(row["decision"], judge_id)
            require(check["outcome"] != "PASS" or row["decision"]["action"] == "ABSTAIN", "unexplained review abstention")
            expected_proposals.append(asdict(proposal))
        require(sorted(expected_proposals, key=lambda p: p["chunk_id"]) == sorted(proposals, key=lambda p: p["chunk_id"]), "review proposal accounting mismatch")
        reconstructed = ReviewState(**state)
        require(state["final_response"] == render(reconstructed) == (path / "response.md").read_text(encoding="utf-8"), "review response mismatch")
    finally:
        memory.close()
    changes = {}
    if current_library is not None:
        with Library(current_library, read_only=True) as live:
            live.audit()
            for packet in packets:
                changes.update(live.verify_packet(packet)["current_source_status"])
    return {"outcome": "PASS", "scope": "snapshot retrieval, exact source attribution, ledger and final response replay",
            "excerpts": len(state["excerpts"]), "verified_factual_claims": 0,
            "current_source_status": changes, "current_library_checked": current_library is not None}

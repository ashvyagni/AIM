"""Explicit reviewed subset export; originals and rejected decisions stay intact."""
from pathlib import Path

from .audit_contracts import fields, seal, text, verify
from .corpus import Corpus, FIELDS, SCHEMA as CORPUS_SCHEMA, intake, read_json, require
from .corpus_review import validate_audit
from .tracking import Run, write_json

SCHEMA = "aim-corpus-review-decisions-v1"
RELEASE = "aim-reviewed-corpus-release-v1"


def review_template(report):
    return {"schema": SCHEMA, "corpus_hash": report["corpus_hash"], "audit_hash": report["record_hash"],
            "purpose": None, "reviewer": None, "basis": None,
            "documents": [{"id": row["id"], "sha256": row["sha256"], "decision": "unreviewed", "reason": None,
                           "rights_evidence": None, "privacy_evidence": None} for row in report["documents"]]}


def validate_review(review, report):
    verify(review, SCHEMA)
    fields(review, {"schema", "corpus_hash", "audit_hash", "purpose", "reviewer", "basis", "documents", "record_hash"}, "review")
    require(review["corpus_hash"] == report["corpus_hash"] and review["audit_hash"] == report["record_hash"], "review must bind exact corpus and audit")
    require(review["purpose"] in {"engineering_fixture", "production_candidate"}, "review purpose must be explicit")
    text(review["reviewer"], "reviewer")
    text(review["basis"], "review basis")
    require(isinstance(review["documents"], list) and len(review["documents"]) == len(report["documents"]), "review must cover every document")
    expected = {r["id"]: r["sha256"] for r in report["documents"]}
    seen = set()
    for row in review["documents"]:
        fields(row, {"id", "sha256", "decision", "reason", "rights_evidence", "privacy_evidence"}, "document review")
        require(isinstance(row["id"], str) and row["id"] not in seen and expected.get(row["id"]) == row["sha256"], "review document missing/duplicate/hash mismatch")
        seen.add(row["id"])
        require(row["decision"] in {"accept", "quarantine", "unreviewed"}, "invalid review decision")
        if row["decision"] != "unreviewed":
            text(row["reason"], "decision reason")
        elif row["reason"] is not None:
            text(row["reason"], "decision reason")
        for key in ("rights_evidence", "privacy_evidence"):
            if row["decision"] == "accept" or row[key] is not None:
                text(row[key], key)
    return review


def gate(corpus, report, review):
    validate_audit(report, corpus)
    validate_review(review, report)
    accepted = {r["id"] for r in review["documents"] if r["decision"] == "accept"}
    blockers = ["unreviewed:"+r["id"] for r in review["documents"] if r["decision"] == "unreviewed"]
    selected_splits = {r["split"] for r in report["documents"] if r["id"] in accepted}
    blockers += ["missing_accepted_split:"+s for s in ("train", "validation") if s not in selected_splits]
    blockers += ["accepted_cross_split_overlap:"+p["left"]+":"+p["right"] for p in report["pairs"]
                 if p["cross_split"] and {p["left"], p["right"]} <= accepted]
    return seal({"schema": "aim-corpus-release-gate-v1", "corpus_hash": corpus.fingerprint, "audit_hash": report["record_hash"],
                 "review_hash": review["record_hash"], "accepted": sorted(accepted),
                 "quarantined": sorted(r["id"] for r in review["documents"] if r["decision"] == "quarantine"),
                 "blockers": sorted(blockers), "status": "BLOCKED" if blockers else "READY_FOR_REVIEWED_SUBSET_EXPORT",
                 "purpose": review["purpose"], "scope": "declared human/operator decisions; lexical gate only; not legal/privacy certification or training-scale approval"})


def export_subset(corpus_path, audit_path, review_path, runs):
    inputs = [Path(corpus_path), Path(audit_path), Path(review_path)]
    with Run(Path(runs), "corpus-reviewed-export", {"scope": "new immutable subset; original corpus unchanged"}, inputs) as run:
        corpus = Corpus(corpus_path)
        report, review = read_json(audit_path, 16*1024*1024), read_json(review_path)
        result = gate(corpus, report, review)
        write_json(run.path/"gate.json", result)
        write_json(run.path/"review.json", review)
        require(not result["blockers"], "reviewed export blocked; inspect retained gate.json")
        source = run.path/"source"
        source.mkdir()
        docs = []
        for row in corpus.index["documents"]:
            if row["id"] not in result["accepted"]:
                continue
            raw = corpus.text(row).encode("utf-8")
            relative = row["id"]+".txt"
            with (source/relative).open("xb") as stream:
                stream.write(raw)
            docs.append({**{k: row[k] for k in FIELDS}, "path": relative})
        manifest = source/"manifest.json"
        write_json(manifest, {"schema": CORPUS_SCHEMA, "version": "reviewed-subset-"+review["record_hash"][:16], "documents": docs})
        child = intake(manifest, run.path/"intake")
        released = Corpus(child)
        write_json(run.path/"release.json", seal({"schema": RELEASE, "parent_corpus_hash": corpus.fingerprint,
                   "audit_hash": report["record_hash"], "review_hash": review["record_hash"], "gate_hash": result["record_hash"],
                   "released_corpus_hash": released.fingerprint, "corpus_path": str(child.relative_to(run.path)),
                   "accepted": result["accepted"], "quarantined": result["quarantined"], "purpose": review["purpose"],
                   "scope": "copied original accepted bytes; no permission inferred, text rewritten, or original record deleted"}))
    return run.path/"release.json"


def audit_release(path, parent_path, audit_path):
    """Recompute decisions and verify every exported object and lineage field."""
    path = Path(path)
    release = read_json(path)
    verify(release, RELEASE)
    fields(release, {"schema", "parent_corpus_hash", "audit_hash", "review_hash", "gate_hash", "released_corpus_hash",
                     "corpus_path", "accepted", "quarantined", "purpose", "scope", "record_hash"}, "release")
    relative = Path(release["corpus_path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "release corpus path escapes export")
    current = path.parent
    for part in relative.parts:
        current = current/part
        require(not current.is_symlink(), "release path contains symlink")
    parent = Corpus(parent_path)
    report = read_json(audit_path, 16*1024*1024)
    review = read_json(path.parent/"review.json")
    result = gate(parent, report, review)
    require(not result["blockers"], "release review no longer passes")
    require(release["parent_corpus_hash"] == parent.fingerprint and release["audit_hash"] == report["record_hash"] and
            release["review_hash"] == review["record_hash"] and release["gate_hash"] == result["record_hash"], "release lineage mismatch")
    require(release["accepted"] == result["accepted"] and release["quarantined"] == result["quarantined"] and release["purpose"] == review["purpose"], "release decision mismatch")
    corpus = Corpus(current)
    require(corpus.fingerprint == release["released_corpus_hash"], "released corpus identity mismatch")
    require(sorted(r["id"] for r in corpus.index["documents"]) == result["accepted"], "exported membership differs")
    original = {r["id"]: r for r in parent.index["documents"]}
    for row in corpus.index["documents"]:
        prior = original[row["id"]]
        require({k: v for k, v in row.items() if k != "path"} == {k: v for k, v in prior.items() if k != "path"}, "export changed original metadata")
        require(corpus.text(row) == parent.text(prior), "export changed original bytes")
    return {"release_hash": release["record_hash"], "documents": len(original), "accepted": len(result["accepted"]),
            "quarantined": len(result["quarantined"]), "original_bytes_and_metadata_preserved": True, "gate_replayed": True}

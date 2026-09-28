"""Tracked local-library operations; no implicit scans, downloads or permissions."""
import argparse
from pathlib import Path

from .corpus import local_path, read_json, require
from .evidence_review import EvidenceReviewController, audit_review
from .library import Library, MAX_DOCUMENT_BYTES, MAX_TOTAL_BYTES, bounded_text, identity
from .tracking import Run, digest, write_json

MANIFEST = "aim-library-intake-v1"
FIELDS = {"path", "sha256", "uri", "version", "title", "rights", "retrieval_allowed", "privacy_review"}


def load_documents(manifest_path):
    path = Path(manifest_path)
    manifest = read_json(path)
    require(isinstance(manifest, dict) and set(manifest) == {"schema", "documents", "chunk_size", "overlap"}
            and manifest["schema"] == MANIFEST, "invalid library intake manifest")
    rows = manifest["documents"]
    require(isinstance(rows, list) and 1 <= len(rows) <= 100, "intake needs 1..100 documents")
    documents, total = [], 0
    for row in rows:
        require(isinstance(row, dict) and set(row) == FIELDS, "invalid library intake document fields")
        require(row["retrieval_allowed"] is True, "explicit retrieval permission declaration required")
        review = row["privacy_review"]
        require(isinstance(review, dict) and set(review) == {"status", "reviewer", "basis"}
                and review["status"] == "approved", "privacy review declaration required")
        for name in ("reviewer", "basis"):
            bounded_text(review[name], "privacy " + name)
        bounded_text(row["path"], "document path")
        source = local_path(path.parent, row["path"])
        require(source.suffix.lower() in {".txt", ".md"}, "initial library parser supports only .txt and .md")
        require(0 < source.stat().st_size <= MAX_DOCUMENT_BYTES, "document size outside library bounds")
        raw = source.read_bytes()
        require(0 < len(raw) <= MAX_DOCUMENT_BYTES and digest(raw) == row["sha256"], "intake source hash or size mismatch")
        total += len(raw)
        require(total <= MAX_TOTAL_BYTES, "intake batch byte budget exceeded")
        document = {"text": raw.decode("utf-8", errors="strict"),
                    **{k: row[k] for k in ("uri", "version", "title", "rights")},
                    "chunk_size": manifest["chunk_size"], "overlap": manifest["overlap"]}
        identity(**document)
        documents.append(document)
    return manifest, documents


def ingest_manifest(library_path, manifest_path, runs):
    with Run(Path(runs), "library-intake", {"schema": MANIFEST, "library_path": str(Path(library_path).resolve()),
             "permission_scope": "owner declarations only; no training permission inferred"}, [Path(manifest_path)]) as run:
        manifest, documents = load_documents(manifest_path)
        write_json(run.path / "intake-manifest.json", manifest)
        with Library(library_path) as library:
            before = library.audit()
            write_json(run.path / "before.json", before)
            ids = library.ingest_documents(documents)
            after = library.audit()
            write_json(run.path / "receipt.json", {"source_ids": ids, "manifest_hash": digest(manifest),
                       "before": before["snapshot"], "after": after["snapshot"],
                       "documents": [{"source_id": sid, "sha256": digest(d["text"].encode())} for sid, d in zip(ids, documents)]})
            write_json(run.path / "after.json", after)
    return run.path


def main(argv=None):
    parser = argparse.ArgumentParser(description="AIM versioned local evidence library")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "ingest", "search", "retire", "audit", "review", "replay", "list", "history", "source"):
        command = commands.add_parser(name)
        command.add_argument("--library", type=Path, required=True)
        command.add_argument("--runs", type=Path, default=Path("runs"))
        if name == "ingest":
            command.add_argument("--manifest", type=Path, required=True)
        if name in {"search", "review"}:
            command.add_argument("--question", required=True)
        if name == "search":
            command.add_argument("--limit", type=int, default=8)
            command.add_argument("--per-source", type=int, default=2)
        if name == "retire":
            command.add_argument("--source-id", required=True)
            command.add_argument("--status", choices=["WITHDRAWN", "RETRACTED"], required=True)
            command.add_argument("--reason", required=True)
            command.add_argument("--actor", required=True)
        if name == "replay":
            command.add_argument("--review-run", type=Path, required=True)
        if name in {"history", "source"}:
            command.add_argument("--source-id", required=name == "source")
    args = parser.parse_args(argv)
    if args.command == "ingest":
        result = ingest_manifest(args.library, args.manifest, args.runs)
    elif args.command == "review":
        result = EvidenceReviewController().run(args.question, args.library, args.runs)
    else:
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        with Run(args.runs, "library-" + args.command, config) as run:
            if args.command == "replay":
                write_json(run.path / "audit.json", audit_review(args.review_run, args.library))
            else:
                with Library(args.library, create=args.command == "init", read_only=args.command in {"search", "audit", "list", "history", "source"}) as library:
                    if args.command == "retire":
                        write_json(run.path / "before.json", library.audit())
                        library.retire(args.source_id, status=args.status, reason=args.reason, actor=args.actor)
                    elif args.command == "search":
                        library.backup(run.path / "library.sqlite")
                        with Library(run.path / "library.sqlite", read_only=True) as snapshot:
                            packet = snapshot.search(args.question, limit=args.limit, per_source=args.per_source)
                            write_json(run.path / "retrieval.json", packet)
                            write_json(run.path / "verification.json", snapshot.verify_packet(packet))
                    elif args.command == "list":
                        write_json(run.path / "inventory.json", library.inventory())
                    elif args.command == "history":
                        write_json(run.path / "history.json", library.history(args.source_id))
                    elif args.command == "source":
                        write_json(run.path / "source.json", library.source(args.source_id))
                    write_json(run.path / "audit.json", library.audit())
        result = run.path
    print(result)


if __name__ == "__main__":
    main()

"""Local, declared-rights corpus intake. No downloads or inferred permission."""
import json
import re
import unicodedata
from pathlib import Path

from .contracts import ContractError
from .tracking import Run, digest, write_json

SCHEMA = "aim-corpus-manifest-v1"
INDEX = "aim-corpus-index-v1"
MAX_DOCUMENT_BYTES = 262144
MAX_TOTAL_BYTES = 16*1024*1024
SPLITS = {"train", "validation", "test"}
FIELDS = {"id", "path", "sha256", "split", "group", "source_uri", "source_version", "content_type", "rights", "privacy_review"}


def require(condition, message):
    if not condition:
        raise ContractError(message)


def read_json(path, limit=4*1024*1024):
    path = Path(path)
    require(path.stat().st_size <= limit, "JSON artifact exceeds size budget")
    try:
        def unique(pairs):
            out = {}
            for key, value in pairs:
                require(key not in out, "duplicate JSON key")
                out[key] = value
            return out
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ContractError("invalid JSON artifact") from exc


def validate_document(row):
    require(isinstance(row, dict) and set(row) == FIELDS, "invalid corpus document fields")
    for key in FIELDS - {"rights", "privacy_review"}:
        require(isinstance(row[key], str) and 1 <= len(row[key]) <= 2048, "missing corpus text metadata")
    require(re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", row["id"]) is not None, "invalid document ID")
    require(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None, "invalid content SHA-256")
    require(row["split"] in SPLITS, "invalid corpus split")
    require(row["content_type"] in {"prose", "code", "math", "unicode"}, "unrecognized corpus content type")
    rights = row["rights"]
    require(isinstance(rights, dict) and set(rights) == {"license", "training_allowed", "basis", "reviewer"}, "explicit rights declaration required")
    require(rights["training_allowed"] is True and all(isinstance(rights[k], str) and 0 < len(rights[k]) <= 2048 for k in ("license", "basis", "reviewer")), "training permission must be explicitly declared")
    review = row["privacy_review"]
    require(isinstance(review, dict) and set(review) == {"status", "basis", "reviewer"} and review["status"] == "approved" and
            all(isinstance(review[k], str) and 0 < len(review[k]) <= 2048 for k in ("basis", "reviewer")), "privacy review declaration required")


def local_path(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "corpus paths must stay within the manifest directory")
    path = root
    for part in relative.parts:
        path = path/part
        require(not path.is_symlink(), "corpus symlinks are not supported")
    require(path.resolve().is_relative_to(root.resolve()) and path.is_file(), "corpus source must be a local regular file")
    return path


def intake(manifest_path, runs):
    manifest_path = Path(manifest_path)
    with Run(Path(runs), "corpus-intake", {"schema": SCHEMA, "permission_scope": "declarations, not independent legal or privacy certification"}, [manifest_path]) as run:
        manifest = read_json(manifest_path)
        require(isinstance(manifest, dict) and set(manifest) == {"schema", "version", "documents"} and manifest["schema"] == SCHEMA,
                "invalid corpus manifest schema")
        require(isinstance(manifest["version"], str) and 0 < len(manifest["version"]) <= 128, "corpus version required")
        rows = manifest["documents"]
        require(isinstance(rows, list) and 1 <= len(rows) <= 10000, "corpus needs 1..10000 documents")
        objects = run.path/"objects"
        objects.mkdir()
        ids, hashes, normalized, groups, records = set(), set(), set(), {}, []
        total = 0
        for row in rows:
            validate_document(row)
            require(row["id"] not in ids, "duplicate document ID")
            require(row["group"] not in groups or groups[row["group"]] == row["split"], "document family leaks across splits")
            source = local_path(manifest_path.parent, row["path"])
            require(0 < source.stat().st_size <= MAX_DOCUMENT_BYTES, "document size outside intake bounds")
            raw = source.read_bytes()
            require(0 < len(raw) <= MAX_DOCUMENT_BYTES and digest(raw) == row["sha256"], "document content hash/size mismatch")
            try:
                text = raw.decode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise ContractError("documents must be strict UTF-8") from exc
            require(bool(text.strip()) and "\x00" not in text, "empty or NUL-containing document")
            norm = digest(" ".join(unicodedata.normalize("NFC", text).split()).encode())
            require(row["sha256"] not in hashes and norm not in normalized, "exact or normalized duplicate; intake rejected without silent dropping")
            total += len(raw)
            require(total <= MAX_TOTAL_BYTES, "miniature corpus total-byte budget exceeded")
            with (objects/row["sha256"]).open("xb") as stream:
                stream.write(raw)
            ids.add(row["id"])
            hashes.add(row["sha256"])
            normalized.add(norm)
            groups[row["group"]] = row["split"]
            records.append({**row, "bytes": len(raw), "characters": len(text), "normalized_sha256": norm})
            run.event("DOCUMENT_ACCEPTED", {"id": row["id"], "sha256": row["sha256"], "split": row["split"]})
        require({r["split"] for r in records} >= {"train", "validation"}, "train and validation documents required")
        index = {"schema": INDEX, "version": manifest["version"], "manifest_hash": digest(manifest),
                 "documents": sorted(records, key=lambda r: r["id"]), "total_bytes": total,
                 "normalization": "NFC + whitespace collapse for duplicate detection only; stored training bytes unchanged"}
        index["corpus_hash"] = digest(index)
        write_json(run.path/"source-manifest.json", manifest)
        write_json(run.path/"corpus.json", index)
    return run.path/"corpus.json"


class Corpus:
    def __init__(self, path):
        self.path = Path(path)
        self.index = read_json(self.path, limit=16*1024*1024)
        require(isinstance(self.index, dict) and self.index.get("schema") == INDEX, "invalid corpus index")
        self.fingerprint = self.index.get("corpus_hash")
        require(self.fingerprint == digest({k: v for k, v in self.index.items() if k != "corpus_hash"}), "corpus index hash mismatch")
        rows = self.index.get("documents")
        require(isinstance(rows, list) and 1 <= len(rows) <= 10000, "invalid corpus index document list")
        groups, ids, hashes, normalized = {}, set(), set(), set()
        for row in rows:
            require(isinstance(row, dict) and set(row) == FIELDS | {"bytes", "characters", "normalized_sha256"}, "invalid indexed document")
            validate_document({k: row[k] for k in FIELDS})
            require(type(row["bytes"]) is int and 0 < row["bytes"] <= MAX_DOCUMENT_BYTES, "invalid indexed document size")
            require(type(row["characters"]) is int and 0 < row["characters"] <= row["bytes"] and isinstance(row["normalized_sha256"], str) and
                    re.fullmatch(r"[0-9a-f]{64}", row["normalized_sha256"]) is not None, "invalid indexed text metadata")
            require(row["id"] not in ids and row["sha256"] not in hashes and row["normalized_sha256"] not in normalized, "duplicate in corpus index")
            require(row["group"] not in groups or groups[row["group"]] == row["split"], "indexed split leakage")
            ids.add(row["id"]); hashes.add(row["sha256"]); normalized.add(row["normalized_sha256"])
            groups[row["group"]] = row["split"]
        require(sum(r["bytes"] for r in rows) == self.index["total_bytes"] <= MAX_TOTAL_BYTES, "invalid indexed byte accounting")

    def records(self, split):
        require(split in SPLITS, "invalid requested split")
        return [r for r in self.index["documents"] if r["split"] == split]

    def text(self, row):
        require(row in self.index["documents"], "document is not in this corpus")
        path = local_path(self.path.parent, "objects/"+row["sha256"])
        require(path.stat().st_size == row["bytes"], "stored document size changed")
        raw = path.read_bytes()
        require(digest(raw) == row["sha256"], "stored document hash mismatch")
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ContractError("stored document is not UTF-8") from exc
        require(len(text) == row["characters"] and digest(" ".join(unicodedata.normalize("NFC", text).split()).encode()) == row["normalized_sha256"], "stored text metadata mismatch")
        return text

    def documents(self, split):
        for row in self.records(split):
            yield row, self.text(row)

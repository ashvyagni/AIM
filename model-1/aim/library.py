"""Versioned local evidence library; retrieval is not factual verification.

All original text, indexes and lifecycle events commit in one SQLite transaction.
Offsets are Python Unicode character offsets into unchanged UTF-8 originals.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .contracts import ContractError
from .corpus import require
from .tracking import canonical, digest

SCHEMA = "aim-evidence-library-v1"
PACKET = "aim-retrieval-packet-v1"
RANKER = "unicode-token-coverage-v1"
MAX_DOCUMENT_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_VERSIONS = 5000
MAX_EVENTS = 20000
ZERO = "0" * 64


def bounded_text(value, label, maximum=2048):
    require(isinstance(value, str) and 0 < len(value) <= maximum and bool(value.strip())
            and "\x00" not in value, f"invalid {label}")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ContractError(f"invalid UTF-8 {label}") from exc
    return value


def terms(text):
    return set(re.findall(r"\w+", text.casefold()))


def chunk_spans(text, size=1024, overlap=128):
    require(type(size) is int and 64 <= size <= 4096, "chunk size must be 64..4096 characters")
    require(type(overlap) is int and 0 <= overlap <= size // 2, "chunk overlap outside bounds")
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        yield start, end
        if end == len(text):
            break
        start = end - overlap


def identity(text, *, uri, version, title, rights, chunk_size=1024, overlap=128):
    bounded_text(text, "document", MAX_DOCUMENT_BYTES)
    raw = text.encode("utf-8")
    require(len(raw) <= MAX_DOCUMENT_BYTES, "document byte budget exceeded")
    for label, value in (("URI", uri), ("version", version), ("title", title), ("rights", rights)):
        bounded_text(value, label)
    list(chunk_spans("x", chunk_size, overlap))
    return {"uri": uri, "version": version, "title": title, "rights": rights,
            "sha256": digest(raw), "bytes": len(raw), "characters": len(text),
            "chunk_size": chunk_size, "overlap": overlap, "parser": "unchanged-utf8-v1"}


class Library:
    """One local file, bounded corpus, serial writers, snapshot readers.

    This database must reside on a local filesystem. It is not a cluster service.
    Schema changes require an explicit future migration; unknown versions fail.
    """

    def __init__(self, path, *, create=False, read_only=False, timeout=5.0):
        self.path = Path(path)
        self.read_only = read_only
        require(not (create and read_only), "cannot create a read-only library")
        require(not self.path.is_symlink(), "library file cannot be a symlink")
        require(create or self.path.is_file(), "library does not exist; initialize explicitly")
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "ro" if read_only else ("rwc" if create else "rw")
        self.db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=" + mode,
                                  uri=True, timeout=timeout, isolation_level=None)
        self.db.execute("PRAGMA foreign_keys=ON")
        if read_only:
            self.db.execute("PRAGMA query_only=ON")
        else:
            self.db.execute("PRAGMA synchronous=FULL")
        try:
            if create:
                with self.transaction(write=True):
                    tables = self.db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                    if not tables:
                        self._initialize()
            require(self.db.execute("PRAGMA user_version").fetchone()[0] == 1, "unsupported library schema version")
            rows = self.db.execute("SELECT record FROM metadata").fetchall()
            require(len(rows) == 1, "invalid library metadata")
            self.metadata = json.loads(rows[0][0])
            require(set(self.metadata) == {"schema", "library_id", "ranker"}
                    and self.metadata["schema"] == SCHEMA and self.metadata["ranker"] == RANKER
                    and re.fullmatch(r"[0-9a-f]{32}", self.metadata["library_id"]), "invalid library identity")
        except BaseException:
            self.db.close()
            raise

    def _initialize(self):
        statements = [
            "CREATE TABLE metadata(record TEXT NOT NULL)",
            "CREATE TABLE sources(id TEXT PRIMARY KEY, uri TEXT NOT NULL, version TEXT NOT NULL, metadata TEXT NOT NULL, text TEXT NOT NULL, bytes INTEGER NOT NULL CHECK(bytes>0), UNIQUE(uri,version))",
            "CREATE TABLE chunks(id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id), start INTEGER NOT NULL, end INTEGER NOT NULL, quote_hash TEXT NOT NULL, CHECK(start>=0 AND end>start), UNIQUE(source_id,start,end))",
            "CREATE TABLE postings(term TEXT NOT NULL, chunk_id TEXT NOT NULL REFERENCES chunks(id), PRIMARY KEY(term,chunk_id))",
            "CREATE INDEX postings_chunk ON postings(chunk_id)",
            "CREATE INDEX chunks_source ON chunks(source_id)",
            "CREATE TABLE events(seq INTEGER PRIMARY KEY, previous TEXT NOT NULL, sha256 TEXT NOT NULL, record TEXT NOT NULL)",
        ]
        for statement in statements:
            self.db.execute(statement)
        for table in ("metadata", "sources", "chunks", "postings", "events"):
            for operation in ("UPDATE", "DELETE"):
                self.db.execute(f"CREATE TRIGGER {table}_no_{operation.lower()} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'immutable library record'); END")
        self.db.execute("INSERT INTO metadata VALUES(?)", (canonical({"schema": SCHEMA, "library_id": uuid.uuid4().hex, "ranker": RANKER}),))
        self.db.execute("PRAGMA user_version=1")

    @contextmanager
    def transaction(self, *, write=False):
        require(not write or not self.read_only, "library is read-only")
        require(not self.db.in_transaction, "nested library transactions are unsupported")
        try:
            self.db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.db.close()

    def _history(self, through=None):
        rows = self.db.execute("SELECT seq,previous,sha256,record FROM events ORDER BY seq").fetchall()
        require(len(rows) <= MAX_EVENTS, "library event budget exceeded")
        if through is None:
            through = len(rows)
        require(type(through) is int and 0 <= through <= len(rows), "unknown library snapshot")
        previous, states, active, snapshot_hash = digest(self.metadata), {}, {}, digest(self.metadata)
        historical = {}
        for expected, (seq, prev, sha, raw) in enumerate(rows, 1):
            require(seq == expected and prev == previous and digest((prev + raw).encode()) == sha, "library ledger hash mismatch")
            event = json.loads(raw)
            require(set(event) == {"seq", "kind", "payload", "time_ns"} and event["seq"] == seq
                    and type(event["time_ns"]) is int and event["time_ns"] > 0 and canonical(event) == raw,
                    "invalid library event")
            p = event["payload"]
            if event["kind"] == "INGEST":
                require(set(p) == {"source_id", "uri", "supersedes"} and p["source_id"] not in states
                        and p["supersedes"] == active.get(p["uri"]), "invalid source revision event")
                if p["supersedes"]:
                    states[p["supersedes"]] = "SUPERSEDED"
                states[p["source_id"]] = "ACTIVE"
                active[p["uri"]] = p["source_id"]
            elif event["kind"] == "RETIRE":
                require(set(p) == {"source_id", "uri", "status", "reason", "actor"}
                        and p["source_id"] in states and p["status"] in {"WITHDRAWN", "RETRACTED"}
                        and states[p["source_id"]] in {"ACTIVE", "SUPERSEDED"}, "invalid retirement event")
                bounded_text(p["reason"], "retirement reason")
                bounded_text(p["actor"], "retirement actor")
                states[p["source_id"]] = p["status"]
                if active.get(p["uri"]) == p["source_id"]:
                    del active[p["uri"]]
            else:
                raise ContractError("unknown library event kind")
            source = self.db.execute("SELECT uri FROM sources WHERE id=?", (p["source_id"],)).fetchone()
            require(source is not None and source[0] == p["uri"], "event source binding mismatch")
            previous = sha
            if seq == through:
                historical, snapshot_hash = dict(states), sha
        snapshot = {"library_id": self.metadata["library_id"], "sequence": through, "sha256": snapshot_hash}
        return historical, snapshot, states

    def _append(self, kind, payload):
        row = self.db.execute("SELECT seq,sha256 FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        seq, previous = (row[0] + 1, row[1]) if row else (1, digest(self.metadata))
        require(seq <= MAX_EVENTS, "library event budget exceeded")
        raw = canonical({"seq": seq, "kind": kind, "payload": payload, "time_ns": time.time_ns()})
        self.db.execute("INSERT INTO events VALUES(?,?,?,?)", (seq, previous, digest((previous + raw).encode()), raw))

    def ingest(self, text, **metadata):
        meta = identity(text, **metadata)
        with self.transaction(write=True):
            return self._ingest(text, meta)

    def ingest_documents(self, documents):
        """All-or-nothing import of up to 100 explicitly supplied source versions."""
        require(isinstance(documents, list) and 1 <= len(documents) <= 100, "batch requires 1..100 documents")
        prepared = []
        for document in documents:
            require(isinstance(document, dict) and "text" in document, "invalid document batch record")
            meta = identity(document["text"], **{k: v for k, v in document.items() if k != "text"})
            prepared.append((document["text"], meta))
        require(len({m["uri"] for _, m in prepared}) == len(prepared), "one version per URI allowed in an import batch")
        require(sum(m["bytes"] for _, m in prepared) <= MAX_TOTAL_BYTES, "batch byte budget exceeded")
        with self.transaction(write=True):
            return [self._ingest(text, meta) for text, meta in prepared]

    def _ingest(self, text, meta):
        sid = "lib-src-" + digest(meta)
        states, _, _ = self._history()
        existing = self.db.execute("SELECT id FROM sources WHERE uri=? AND version=?", (meta["uri"], meta["version"])).fetchone()
        if existing:
            require(existing[0] == sid, "source version already exists with different content or metadata")
            self._source(sid)
            require(sid in states, "source missing ingestion event")
            return sid
        count, total = self.db.execute("SELECT count(*),coalesce(sum(bytes),0) FROM sources").fetchone()
        require(count < MAX_VERSIONS and total + meta["bytes"] <= MAX_TOTAL_BYTES, "library capacity exceeded")
        current = [s for s, state in states.items() if state == "ACTIVE" and
                   self.db.execute("SELECT uri FROM sources WHERE id=?", (s,)).fetchone()[0] == meta["uri"]]
        require(len(current) <= 1, "multiple active versions for one URI")
        self.db.execute("INSERT INTO sources VALUES(?,?,?,?,?,?)", (sid, meta["uri"], meta["version"], canonical(meta), text, meta["bytes"]))
        for start, end in chunk_spans(text, meta["chunk_size"], meta["overlap"]):
            cid = "lib-chunk-" + digest([sid, start, end])
            self.db.execute("INSERT INTO chunks VALUES(?,?,?,?,?)", (cid, sid, start, end, digest(text[start:end].encode())))
            self.db.executemany("INSERT INTO postings VALUES(?,?)", [(term, cid) for term in sorted(terms(meta["title"] + " " + text[start:end]))])
        self._append("INGEST", {"source_id": sid, "uri": meta["uri"], "supersedes": current[0] if current else None})
        return sid

    def retire(self, source_id, *, status, reason, actor):
        require(status in {"WITHDRAWN", "RETRACTED"}, "unsupported retirement status")
        bounded_text(reason, "retirement reason")
        bounded_text(actor, "retirement actor")
        with self.transaction(write=True):
            states, _, _ = self._history()
            require(states.get(source_id) in {"ACTIVE", "SUPERSEDED"}, "source cannot be retired from its current state")
            _, meta = self._source(source_id)
            self._append("RETIRE", {"source_id": source_id, "uri": meta["uri"], "status": status, "reason": reason, "actor": actor})

    def _source(self, source_id):
        row = self.db.execute("SELECT uri,version,metadata,text,bytes FROM sources WHERE id=?", (source_id,)).fetchone()
        require(row is not None, "unknown library source")
        meta = json.loads(row[2])
        expected = identity(row[3], **{k: meta[k] for k in ("uri", "version", "title", "rights", "chunk_size", "overlap")})
        require(meta == expected and source_id == "lib-src-" + digest(meta) and row[0] == meta["uri"]
                and row[1] == meta["version"] and row[4] == meta["bytes"], "library source integrity failure")
        return row[3], meta

    def source(self, source_id):
        with self.transaction():
            states, _, _ = self._history()
            require(source_id in states, "source missing ingestion event")
            text, meta = self._source(source_id)
            return {"id": source_id, "metadata": meta, "text": text, "status": states[source_id]}

    def inventory(self):
        with self.transaction():
            states, snapshot, _ = self._history()
            sources = []
            for sid in sorted(states):
                _, meta = self._source(sid)
                sources.append({"source_id": sid, "status": states[sid], "metadata": meta})
            return {"snapshot": snapshot, "sources": sources}

    def history(self, source_id=None):
        with self.transaction():
            states, snapshot, _ = self._history()
            require(source_id is None or source_id in states, "unknown source history")
            events = [json.loads(row[0]) for row in self.db.execute("SELECT record FROM events ORDER BY seq")]
            if source_id is not None:
                events = [e for e in events if e["payload"]["source_id"] == source_id or e["payload"].get("supersedes") == source_id]
            return {"snapshot": snapshot, "events": events}

    def _hit(self, row, query_terms):
        cid, sid, start, end, sha = row
        text, meta = self._source(sid)
        require(type(start) is int and type(end) is int and (start, end) in set(chunk_spans(text, meta["chunk_size"], meta["overlap"]))
                and cid == "lib-chunk-" + digest([sid, start, end]), "invalid library chunk locator")
        quote = text[start:end]
        require(digest(quote.encode()) == sha, "library quote integrity failure")
        matched = sorted(query_terms & terms(meta["title"] + " " + quote))
        return {"chunk_id": cid, "source_id": sid, "metadata": meta, "start": start, "end": end,
                "quote": quote, "quote_hash": sha, "matched_terms": matched, "score": len(matched)}

    def _search(self, query, limit, per_source, through=None):
        bounded_text(query, "query", 2048)
        require(type(limit) is int and 1 <= limit <= 32 and type(per_source) is int and 1 <= per_source <= 8, "invalid retrieval limits")
        query_terms = terms(query)
        require(1 <= len(query_terms) <= 64, "query requires 1..64 distinct word tokens")
        states, snapshot, _ = self._history(through)
        placeholders = ",".join("?" for _ in query_terms)
        rows = self.db.execute(f"SELECT c.id,c.source_id,c.start,c.end,c.quote_hash,count(*) AS score FROM postings p JOIN chunks c ON c.id=p.chunk_id WHERE p.term IN ({placeholders}) GROUP BY c.id ORDER BY score DESC,c.source_id,c.start,c.id", tuple(sorted(query_terms)))
        hits, counts = [], {}
        for row in rows:
            sid = row[1]
            if states.get(sid) != "ACTIVE" or counts.get(sid, 0) >= per_source:
                continue
            hit = self._hit(row[:5], query_terms)
            require(hit["score"] == row[5] and hit["score"] > 0, "retrieval index mismatch")
            hits.append(hit)
            counts[sid] = counts.get(sid, 0) + 1
            if len(hits) == limit:
                break
        packet = {"schema": PACKET, "ranker": RANKER, "snapshot": snapshot, "query": query,
                  "limit": limit, "per_source": per_source, "hits": hits,
                  "scope": "lexical matches in declared local sources; no entailment or factuality judgment"}
        return {**packet, "packet_hash": digest(packet)}

    def search(self, query, *, limit=8, per_source=2):
        with self.transaction():
            return self._search(query, limit, per_source)

    def verify_packet(self, packet):
        require(isinstance(packet, dict) and set(packet) == {"schema", "ranker", "snapshot", "query", "limit", "per_source", "hits", "scope", "packet_hash"}, "invalid retrieval packet")
        require(packet["packet_hash"] == digest({k: v for k, v in packet.items() if k != "packet_hash"}), "retrieval packet hash mismatch")
        snapshot = packet["snapshot"]
        require(isinstance(snapshot, dict) and set(snapshot) == {"library_id", "sequence", "sha256"}
                and snapshot["library_id"] == self.metadata["library_id"], "packet belongs to a different library")
        with self.transaction():
            expected = self._search(packet["query"], packet["limit"], packet["per_source"], snapshot["sequence"])
            require(packet == expected, "retrieval packet does not replay exactly")
            states, current, _ = self._history()
            return {"outcome": "PASS", "scope": "historical retrieval and quote integrity only",
                    "current_snapshot": current, "current_source_status": {h["source_id"]: states[h["source_id"]] for h in packet["hits"]}}

    def audit(self):
        with self.transaction():
            require(self.db.execute("SELECT record FROM metadata").fetchall() == [(canonical(self.metadata),)], "library metadata changed")
            require(self.db.execute("PRAGMA integrity_check").fetchall() == [("ok",)], "SQLite integrity check failed")
            require(not self.db.execute("PRAGMA foreign_key_check").fetchall(), "library foreign key failure")
            states, snapshot, _ = self._history()
            ids = [r[0] for r in self.db.execute("SELECT id FROM sources ORDER BY id")]
            require(set(ids) == set(states) and len(ids) <= MAX_VERSIONS, "library source/event inventory mismatch")
            count, total = 0, 0
            for sid in ids:
                text, meta = self._source(sid)
                total += meta["bytes"]
                expected = [("lib-chunk-" + digest([sid, a, b]), sid, a, b, digest(text[a:b].encode()))
                            for a, b in chunk_spans(text, meta["chunk_size"], meta["overlap"])]
                rows = self.db.execute("SELECT id,source_id,start,end,quote_hash FROM chunks WHERE source_id=? ORDER BY start", (sid,)).fetchall()
                require(rows == expected, "library chunk inventory mismatch")
                for cid, _, a, b, _ in rows:
                    actual = {r[0] for r in self.db.execute("SELECT term FROM postings WHERE chunk_id=?", (cid,))}
                    require(actual == terms(meta["title"] + " " + text[a:b]), "library posting inventory mismatch")
                count += len(rows)
            require(total <= MAX_TOTAL_BYTES, "library byte budget exceeded")
            return {"schema": SCHEMA, "snapshot": snapshot, "outcome": "PASS", "versions": len(ids),
                    "chunks": count, "bytes": total, "statuses": {s: sum(v == s for v in states.values())
                    for s in ("ACTIVE", "SUPERSEDED", "WITHDRAWN", "RETRACTED")},
                    "scope": "stored content, offsets, full lexical index, lifecycle and hash-chain consistency; not source truth"}

    def backup(self, destination):
        """Consistent local SQLite snapshot. Refuse to overwrite even an empty file."""
        destination = Path(destination)
        require(not self.db.in_transaction, "cannot back up within a library transaction")
        with destination.open("xb"):
            pass
        target = sqlite3.connect(destination)
        try:
            self.db.backup(target)
        finally:
            target.close()
        with Library(destination, read_only=True) as snapshot:
            return snapshot.audit()

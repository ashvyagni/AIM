"""SQLite provenance ledger + content-addressed originals; no vector-as-truth shortcut."""
from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path

from .contracts import ContractError, Evidence
from .tracking import canonical, digest


class Memory:
    def __init__(self, directory: Path, *, read_only=False):
        directory=Path(directory)
        self.read_only=read_only
        if read_only:
            self.objects=directory/"objects"
            database=directory/"memory.sqlite"
            if not database.is_file() or not self.objects.is_dir():
                raise ContractError("Read-only memory requires an existing store")
            self.db=sqlite3.connect(database.resolve().as_uri()+"?mode=ro",uri=True)
            self.db.execute("PRAGMA query_only=ON")
            return
        directory.mkdir(parents=True, exist_ok=True)
        self.objects = directory / "objects"
        self.objects.mkdir(exist_ok=True)
        self.db = sqlite3.connect(directory / "memory.sqlite")
        self.db.executescript("""
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS sources (
                id TEXT PRIMARY KEY, sha256 TEXT NOT NULL, metadata TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS evidence (
                id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id), record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY, previous TEXT NOT NULL, sha256 TEXT NOT NULL, record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS edges (
                origin TEXT NOT NULL, relation TEXT NOT NULL, target TEXT NOT NULL,
                PRIMARY KEY(origin,relation,target));
            CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
                BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
                BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS sources_no_update BEFORE UPDATE ON sources
                BEGIN SELECT RAISE(ABORT, 'source versions are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS evidence_no_update BEFORE UPDATE ON evidence
                BEGIN SELECT RAISE(ABORT, 'evidence is immutable'); END;
        """)

    def close(self):
        self.db.close()

    def _writable(self):
        if self.read_only: raise ContractError("Memory is read-only")

    def ingest(self, text: str, *, uri: str, version: str, rights: str, title: str, retrievable=True) -> str:
        self._writable()
        raw = text.encode("utf-8")
        sha = digest(raw)
        identity = {"sha256": sha, "uri": uri, "version": version, "rights": rights,
                    "title": title, "retrievable": retrievable}
        sid = "src-" + digest(identity)[:24]
        path = self.objects / sha
        if path.exists():
            if digest(path.read_bytes()) != sha:
                raise ContractError("Original artifact hash mismatch")
        else:
            path.write_bytes(raw)
        metadata = {**identity, "retrieved_at": time.time(), "parser": "utf8-v1"}
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO sources VALUES(?,?,?)", (sid, sha, canonical(metadata)))
        return sid

    def source(self, source_id: str):
        row = self.db.execute("SELECT sha256,metadata FROM sources WHERE id=?", (source_id,)).fetchone()
        if row is None:
            raise ContractError("Unknown source")
        raw = (self.objects / row[0]).read_bytes()
        if digest(raw) != row[0]:
            raise ContractError("Source content was altered")
        return raw.decode("utf-8"), json.loads(row[1])

    def span(self, source_id: str, start: int, end: int) -> Evidence:
        self._writable()
        text, meta = self.source(source_id)
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise ContractError("Invalid source character offsets")
        quote = text[start:end]
        eid = "ev-" + digest([source_id, start, end])[:24]
        ev = Evidence(eid, source_id, meta["sha256"], start, end, quote, digest(quote.encode()), f"chars:{start}:{end}")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO evidence VALUES(?,?,?)", (eid, source_id, canonical(ev)))
        self.edge(eid, "EXTRACTED_FROM", source_id)
        return ev

    def validate(self, evidence: Evidence) -> bool:
        row = self.db.execute("SELECT record FROM evidence WHERE id=?", (evidence.id,)).fetchone()
        if row is None or row[0] != canonical(evidence):
            return False
        text, meta = self.source(evidence.source_id)
        return (meta["sha256"] == evidence.source_hash and text[evidence.start:evidence.end] == evidence.quote
                and digest(evidence.quote.encode()) == evidence.quote_hash)

    def search(self, query: str, limit=5) -> list[Evidence]:
        # Deterministic token-overlap reference retriever; BM25/vector are future adapters.
        terms = set(re.findall(r"\w+", query.casefold()))
        scored = []
        for sid, metadata in self.db.execute("SELECT id,metadata FROM sources ORDER BY id"):
            meta = json.loads(metadata)
            if not meta["retrievable"]:
                continue
            text, _ = self.source(sid)
            tokens = set(re.findall(r"\w+", (meta["title"] + " " + text).casefold()))
            score = len(tokens & terms)
            if score:
                scored.append((-score, sid, text))
        return [self.span(sid, 0, len(text)) for _, sid, text in sorted(scored)[:limit]]

    def edge(self, origin, relation, target):
        self._writable()
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO edges VALUES(?,?,?)", (origin, relation, target))

    def append(self, kind: str, payload: dict):
        self._writable()
        with self.db:
            last = self.db.execute("SELECT seq,sha256 FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            seq, previous = (last[0] + 1, last[1]) if last else (1, "0" * 64)
            record = canonical({"seq": seq, "kind": kind, "time_unix": time.time(), "payload": payload})
            sha = digest((previous + record).encode())
            self.db.execute("INSERT INTO events VALUES(?,?,?,?)", (seq, previous, sha, record))

    def replay(self) -> dict | None:
        previous = "0" * 64
        state = None
        for expected, (seq, prev, sha, record) in enumerate(self.db.execute("SELECT * FROM events ORDER BY seq"), 1):
            if seq != expected or prev != previous or digest((prev + record).encode()) != sha:
                raise ContractError("Event ledger integrity failure")
            event = json.loads(record)
            if event["kind"] == "STATE":
                state = event["payload"]
            previous = sha
        return state

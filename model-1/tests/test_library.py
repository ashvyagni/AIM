import copy
import json
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from aim.contracts import ContractError
from aim.library import Library, chunk_spans
from aim.library_cli import ingest_manifest, load_documents
from aim.tracking import digest, write_json


def document(text="Calibration evidence from an authored example.", **changes):
    return {"text": text, "uri": "fixture:calibration", "version": "1", "title": "Calibration example",
            "rights": "AIM-authored test fixture; retrieval allowed", "chunk_size": 64, "overlap": 8, **changes}


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "library.sqlite"
        self.library = Library(self.path, create=True)

    def tearDown(self):
        self.library.close()
        self.temp.cleanup()

    def test_explicit_creation_and_schema_rejection(self):
        with self.assertRaises(ContractError):
            Library(self.root / "missing.sqlite")
        other = self.root / "unknown.sqlite"
        with sqlite3.connect(other) as db:
            db.execute("PRAGMA user_version=99")
        with self.assertRaises(ContractError):
            Library(other)
        with self.assertRaises(ContractError):
            Library(other, create=True)
        with Library(self.path, create=True) as again:
            self.assertEqual(again.metadata, self.library.metadata)

    def test_empty_snapshot_packet(self):
        packet = self.library.search("calibration")
        self.assertEqual(packet["hits"], [])
        self.assertEqual(packet["snapshot"]["sequence"], 0)
        self.library.ingest(**document())
        self.assertEqual(self.library.verify_packet(packet)["outcome"], "PASS")

    def test_unicode_offsets_roundtrip(self):
        text = "αβ 🧪 calibration e\u0301 line\r\n" * 15
        sid = self.library.ingest(**document(text))
        self.assertEqual(self.library.source(sid)["text"], text)
        hits = self.library.search("calibration", limit=32, per_source=8)["hits"]
        self.assertTrue(hits)
        for hit in hits:
            self.assertEqual(text[hit["start"]:hit["end"]], hit["quote"])
            self.assertEqual(digest(hit["quote"].encode()), hit["quote_hash"])
        covered = set()
        for a, b in chunk_spans(text, 64, 8):
            covered.update(range(a, b))
        self.assertEqual(covered, set(range(len(text))))
        self.library.audit()

    def test_idempotence_and_version_conflict(self):
        sid = self.library.ingest(**document())
        before = self.library.audit()
        self.assertEqual(sid, self.library.ingest(**document()))
        self.assertEqual(before, self.library.audit())
        for field, value in (("text", "changed"), ("title", "changed"), ("rights", "changed"), ("overlap", 4)):
            with self.subTest(field=field), self.assertRaises(ContractError):
                self.library.ingest(**document(**{field: value}))
        self.assertEqual(before, self.library.audit())

    def test_revisions_replay_and_no_reactivation(self):
        old = self.library.ingest(**document())
        packet = self.library.search("calibration")
        new = self.library.ingest(**document("Revised calibration report.", version="2"))
        self.assertEqual(self.library.source(old)["status"], "SUPERSEDED")
        self.assertEqual({h["source_id"] for h in self.library.search("calibration")["hits"]}, {new})
        checked = self.library.verify_packet(packet)
        self.assertEqual(checked["current_source_status"][old], "SUPERSEDED")
        self.library.ingest(**document())
        self.assertEqual(self.library.source(old)["status"], "SUPERSEDED")
        self.assertEqual(self.library.audit()["snapshot"]["sequence"], 2)

    def test_retirement_does_not_resurrect_previous_version(self):
        old = self.library.ingest(**document())
        new = self.library.ingest(**document("Calibration revised.", version="2"))
        packet = self.library.search("calibration")
        self.library.retire(new, status="RETRACTED", reason="fixture correction", actor="fixture-reviewer")
        self.assertEqual(self.library.search("calibration")["hits"], [])
        self.assertEqual(self.library.verify_packet(packet)["current_source_status"][new], "RETRACTED")
        self.assertEqual(self.library.source(old)["status"], "SUPERSEDED")
        with self.assertRaises(ContractError):
            self.library.retire(new, status="WITHDRAWN", reason="again", actor="test")
        self.library.retire(old, status="WITHDRAWN", reason="historical removal", actor="test")
        self.assertEqual(self.library.audit()["statuses"]["WITHDRAWN"], 1)

    def test_new_version_after_retirement_is_explicit(self):
        old = self.library.ingest(**document())
        self.library.retire(old, status="WITHDRAWN", reason="fixture", actor="test")
        self.assertEqual(old, self.library.ingest(**document()))
        self.assertEqual(self.library.search("calibration")["hits"], [])
        new = self.library.ingest(**document(version="2"))
        self.assertEqual(self.library.source(new)["status"], "ACTIVE")

    def test_inventory_and_source_history_include_supersession(self):
        old = self.library.ingest(**document())
        self.library.ingest(**document(version="2"))
        history = self.library.history(old)
        self.assertEqual(len(history["events"]), 2)
        self.assertEqual(history["events"][1]["payload"]["supersedes"], old)
        self.assertEqual(len(self.library.inventory()["sources"]), 2)
        with self.assertRaises(ContractError):
            self.library.history("missing")

    def test_lock_timeout_does_not_write_partial_source(self):
        before = self.library.audit()
        with self.library.transaction(write=True):
            with Library(self.path, timeout=0.01) as other:
                with self.assertRaises(sqlite3.OperationalError):
                    other.ingest(**document())
        self.assertEqual(before, self.library.audit())

    def test_read_transaction_has_stable_snapshot(self):
        self.library.ingest(**document())
        appended = threading.Event()
        def write():
            with Library(self.path) as other:
                original = other._append
                def signal(kind, payload):
                    original(kind, payload)
                    appended.set()
                with patch.object(other, "_append", side_effect=signal):
                    return other.ingest(**document(version="2"))
        with ThreadPoolExecutor(max_workers=1) as pool:
            with self.library.transaction():
                before = self.library._history()[1]
                future = pool.submit(write)
                self.assertTrue(appended.wait(timeout=5))
                self.assertEqual(before, self.library._history()[1])
            future.result(timeout=5)
        self.assertEqual(self.library.audit()["snapshot"]["sequence"], 2)

    def test_batch_conflict_rolls_back_every_document(self):
        self.library.ingest(**document())
        before = self.library.audit()
        with self.assertRaises(ContractError):
            self.library.ingest_documents([document(uri="fixture:new"), document("conflicting content")])
        self.assertEqual(before, self.library.audit())

    def test_injected_failure_rolls_back_index_and_source(self):
        before = self.library.audit()
        with patch.object(self.library, "_append", side_effect=RuntimeError("injected before ledger commit")):
            with self.assertRaises(RuntimeError):
                self.library.ingest(**document())
        self.assertEqual(before, self.library.audit())
        self.assertEqual(self.library.db.execute("SELECT count(*) FROM postings").fetchone()[0], 0)

    def test_immutable_rows(self):
        self.library.ingest(**document())
        for table in ("metadata", "sources", "chunks", "postings", "events"):
            for operation in (f"DELETE FROM {table}", f"UPDATE {table} SET " + {"metadata": "record=record", "sources": "text=text", "chunks": "start=start", "postings": "term=term", "events": "seq=seq"}[table]):
                with self.subTest(sql=operation), self.assertRaises(sqlite3.IntegrityError):
                    self.library.db.execute(operation)
        self.library.audit()

    def test_source_tampering_detected(self):
        sid = self.library.ingest(**document())
        self.library.db.execute("DROP TRIGGER sources_no_update")
        self.library.db.execute("UPDATE sources SET text='altered'")
        with self.assertRaises(ContractError):
            self.library.source(sid)
        with self.assertRaises(ContractError):
            self.library.audit()

    def test_missing_posting_detected_by_full_audit(self):
        self.library.ingest(**document())
        self.library.db.execute("DROP TRIGGER postings_no_delete")
        self.library.db.execute("DELETE FROM postings WHERE term='calibration'")
        with self.assertRaises(ContractError):
            self.library.audit()

    def test_packet_tampering_even_if_rehashed(self):
        self.library.ingest(**document())
        original = self.library.search("calibration")
        for mutate in (lambda p: p["hits"][0].update(quote="invented"), lambda p: p.update(hits=[]),
                       lambda p: p["snapshot"].update(sequence=999), lambda p: p.update(ranker="fake")):
            packet = copy.deepcopy(original)
            mutate(packet)
            packet["packet_hash"] = digest({k: v for k, v in packet.items() if k != "packet_hash"})
            with self.assertRaises(ContractError):
                self.library.verify_packet(packet)

    def test_cross_library_packet_rejected(self):
        self.library.ingest(**document())
        with Library(self.root / "other.sqlite", create=True) as other:
            other.ingest(**document())
            with self.assertRaises(ContractError):
                other.verify_packet(self.library.search("calibration"))

    def test_backup_is_independent_and_exclusive(self):
        self.library.ingest(**document())
        destination = self.root / "backup.sqlite"
        before = self.library.backup(destination)
        self.library.ingest(**document(version="2"))
        with Library(destination, read_only=True) as snapshot:
            self.assertEqual(before, snapshot.audit())
            with self.assertRaises(ContractError):
                snapshot.ingest(**document(version="3"))
        with self.assertRaises(FileExistsError):
            self.library.backup(destination)

    def test_backup_deadline_retains_destination(self):
        self.library.ingest(**document())
        destination = self.root / "interrupted-backup.sqlite"
        with patch("aim.library.time.monotonic", side_effect=[0, 31]):
            with self.assertRaisesRegex(ContractError, "backup deadline"):
                self.library.backup(destination)
        self.assertTrue(destination.exists())
        self.assertEqual(self.library.audit()["versions"], 1)
        with self.assertRaises(FileExistsError):
            self.library.backup(destination)

    def test_backup_rejects_invalid_deadlines(self):
        for value in (True, 0, float("nan"), 61):
            with self.subTest(value=value), self.assertRaises(ContractError):
                self.library.backup(self.root / "bad-backup.sqlite", timeout_seconds=value)
        self.assertFalse((self.root / "bad-backup.sqlite").exists())

    def test_query_ranking_and_source_caps(self):
        one = self.library.ingest(**document("alpha beta " * 60, title="one"))
        two = self.library.ingest(**document("alpha gamma", uri="fixture:two", title="two"))
        packet = self.library.search("alpha beta", limit=8, per_source=1)
        self.assertEqual([h["source_id"] for h in packet["hits"]], [one, two])
        self.assertEqual([h["score"] for h in packet["hits"]], [2, 1])
        self.assertEqual(packet, self.library.search("alpha beta", limit=8, per_source=1))

    def test_input_bounds(self):
        for query in ("", "!!!", "a " * 2049, "\x00bad"):
            with self.subTest(query=query[:10]), self.assertRaises(ContractError):
                self.library.search(query)
        for change in ({"chunk_size": True}, {"overlap": 40}, {"text": "\ud800"}, {"text": "\x00"}, {"rights": ""}):
            with self.subTest(change=change), self.assertRaises(ContractError):
                self.library.ingest(**document(**change))
        with self.assertRaises(ContractError):
            self.library.search("alpha", limit=True)

    def test_parallel_writers_have_one_active_head(self):
        barrier = threading.Barrier(4)
        def write(index):
            with Library(self.path, timeout=10) as library:
                barrier.wait(timeout=10)
                return library.ingest(**document(version=str(index)))
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(write, range(4)))
        self.assertEqual(len(set(ids)), 4)
        audit = self.library.audit()
        self.assertEqual(audit["snapshot"]["sequence"], 4)
        self.assertEqual(audit["statuses"]["ACTIVE"], 1)
        self.assertEqual(audit["statuses"]["SUPERSEDED"], 3)

    def manifest(self):
        raw = b"An authored calibration fixture.\r\n"
        (self.root / "source.md").write_bytes(raw)
        value = {"schema": "aim-library-intake-v1", "chunk_size": 64, "overlap": 8, "documents": [
            {"path": "source.md", "sha256": digest(raw), "uri": "fixture:manifest", "version": "1", "title": "Fixture",
             "rights": "AIM authored fixture", "retrieval_allowed": True,
             "privacy_review": {"status": "approved", "reviewer": "fixture-author", "basis": "synthetic text"}}]}
        path = self.root / "manifest.json"
        write_json(path, value)
        return path, value

    def test_manifest_import_tracks_declarations_and_exact_bytes(self):
        path, value = self.manifest()
        run = ingest_manifest(self.path, path, self.root / "runs")
        receipt = json.loads((run / "receipt.json").read_text())
        self.assertEqual(receipt["manifest_hash"], digest(value))
        self.assertTrue(self.library.source(receipt["source_ids"][0])["text"].endswith("\r\n"))
        self.assertEqual(json.loads((run / "status.json").read_text())["status"], "COMPLETED")

    def test_manifest_failures_retained_without_mutation(self):
        path, value = self.manifest()
        value["documents"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(value))
        with self.assertRaises(ContractError):
            ingest_manifest(self.path, path, self.root / "runs")
        failed = list((self.root / "runs").glob("*/status.json"))
        self.assertEqual(len(failed), 1)
        self.assertEqual(json.loads(failed[0].read_text())["status"], "FAILED")
        self.assertEqual(self.library.audit()["versions"], 0)

    def test_manifest_permission_and_path_rejection(self):
        path, value = self.manifest()
        for field, changed in (("retrieval_allowed", False), ("path", "../source.md"), ("privacy_review", {})):
            variant = copy.deepcopy(value)
            variant["documents"][0][field] = changed
            path.write_text(json.dumps(variant))
            with self.assertRaises(ContractError):
                load_documents(path)


if __name__ == "__main__":
    unittest.main()

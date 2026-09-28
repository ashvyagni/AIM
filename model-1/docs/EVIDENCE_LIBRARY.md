# Shared document library and evidence review

Phase 3F adds a separately versioned local document system. It preserves the approved Researcher → separate Judge → deterministic Controller → independent verifiers → provenance memory architecture. The numerical and symbolic research loops and learned model defaults remain available.

## What this adds

- Explicit intake of local UTF-8 `.txt` and `.md` files with unchanged bytes, SHA-256, source URI/version/title, rights declaration and privacy-review declaration.
- Immutable source versions and overlapping chunks with exact Unicode character offsets.
- Persistent indexed lexical retrieval across independent runs.
- Replacement, withdrawal and retraction events, with historical retrieval replay.
- A separate document Researcher interface, Judge interface, attribution verifier and bounded Controller.
- Per-run SQLite snapshots, original source copies in existing Memory, evidence records, state transitions and traceable Markdown excerpts.
- Read-only integrity auditing, source inventory/history commands, transactional batch intake and retained failures.

This implementation produces **evidence reviews**, not unrestricted scientific answers. The reference Researcher extracts retrieved text. It does not generate or validate novel scientific hypotheses. All quoted assertions remain **UNVERIFIED**, including assertions that sound certain or contain the word VERIFIED. A verifier PASS means the excerpt and its attribution match stored source bytes.

## First use

From `model-1/`, initialize a new library in the ignored runs area:

```sh
.venv/bin/python -m aim.library_cli init --library runs/team-library/library.sqlite
```

Create a manifest next to the files you intend to ingest. Example shape below: replace the file hash and declarations with actual values before running it. A hash must describe the original bytes, including line endings; do not hash normalized text.

```json
{
  "schema": "aim-library-intake-v1",
  "chunk_size": 1024,
  "overlap": 128,
  "documents": [
    {
      "path": "methods.md",
      "sha256": "REPLACE_WITH_SHA256_OF_FILE_BYTES",
      "uri": "project:methods",
      "version": "reviewed-revision-1",
      "title": "Methods note",
      "rights": "Record the actual permission and permitted use",
      "retrieval_allowed": true,
      "privacy_review": {
        "status": "approved",
        "reviewer": "Record the actual reviewer",
        "basis": "Record what was reviewed and why local retrieval is permitted"
      }
    }
  ]
}
```

Paths must be regular files beneath the manifest directory, without `..` or symlink components. There is no recursive scan, web fetch, PDF parser, OCR or implicit license inference. Manifests allow 1–100 documents and one version per URI in a batch. The entire batch commits or rolls back. Review declarations are records supplied by the caller; the software cannot certify them. Retrieval permission does not confer training permission.

```sh
.venv/bin/python -m aim.library_cli ingest --library runs/team-library/library.sqlite --manifest runs/documents/manifest.json
.venv/bin/python -m aim.library_cli list --library runs/team-library/library.sqlite
.venv/bin/python -m aim.library_cli search --library runs/team-library/library.sqlite --question "calibration uncertainty"
.venv/bin/python -m aim.library_cli review --library runs/team-library/library.sqlite --question "calibration uncertainty"
.venv/bin/python -m aim.library_cli audit --library runs/team-library/library.sqlite
```

Every command prints a retained run directory with configuration, source hashes, environment, commit and status. Intake returns source IDs in `receipt.json`. Search writes `retrieval.json`; review writes `response.md`, `state.json`, `proposals.json`, `retrieval.json`, `library.sqlite`, `memory/` and `replay-audit.json`. Failed commands retain status and traceback. A successful ingestion transaction can precede a later run-record I/O failure; inspect the library history and retry the same manifest idempotently rather than assuming the transaction did not commit.

## Versions and retirement

The exact URI identifies a source lineage. A new version of the same URI supersedes its currently active version. Version strings are opaque identifiers, not dates or ordered semantic versions. Intake order explicitly determines the active head. Identical bytes at different URIs remain separate attributed sources; this does not prove independence.

Reimporting an identical URI/version/content/metadata is idempotent. Reusing a URI/version with changed content, title, rights or chunk settings fails. Reimporting an old version never reactivates it. To intentionally publish corrected material after retirement, submit a new version and review declaration.

```sh
.venv/bin/python -m aim.library_cli source --library runs/team-library/library.sqlite --source-id SOURCE_ID
.venv/bin/python -m aim.library_cli history --library runs/team-library/library.sqlite --source-id SOURCE_ID
.venv/bin/python -m aim.library_cli retire --library runs/team-library/library.sqlite --source-id SOURCE_ID --status RETRACTED --reason "Record the actual reason" --actor "Actual reviewer"
.venv/bin/python -m aim.library_cli replay --library runs/team-library/library.sqlite --review-run runs/ACTUAL_REVIEW_RUN
```

Allowed retirement statuses are WITHDRAWN and RETRACTED, from ACTIVE or SUPERSEDED. Retirement is terminal for that version. Retiring the latest version does not reactivate its predecessor. Retired text remains available for historical audits; this is **not an erasure or privacy-deletion service**. Administrative deletion, compaction and retention policies need a separate design.

Replay checks the original run snapshot and reports current statuses from the supplied live library. It does not rewrite the original response. Consumers must examine `current_source_status`; a historical attribution PASS can coexist with a currently RETRACTED source.

## Storage and identity

Schema `aim-evidence-library-v1` / SQLite `user_version=1` has five tables:

| Table | Invariant |
|---|---|
| metadata | One schema, library UUID and ranker identity |
| sources | Immutable original text and metadata; unique URI/version |
| chunks | Immutable original character spans and quote hashes |
| postings | Exact word-token membership for title plus chunk |
| events | Sequential lifecycle events chained by SHA-256 |

Indexes and immutable-row triggers are separate schema objects. Each table rejects UPDATE and DELETE through normal SQL. Foreign keys bind chunks and postings. Source identity hashes include byte hash, metadata, parser and chunk settings. Chunk identity hashes include source ID and start/end positions. Unchanged UTF-8 text is stored inside the same database transaction as chunks, index and event; there is no external-object publication gap.

The initial hash-chain root binds library UUID/schema/ranker. Lifecycle state is replayed from events, without a separately mutable status projection. Full audits recompute original hashes, all chunks, every posting, inventory and event chain, plus SQLite integrity and foreign-key checks. Cheap search validates the chain and returned content; a full audit is required to detect a deliberately deleted index entry. The review Controller performs a full snapshot audit first.

Use a **local filesystem**. Writers take `BEGIN IMMEDIATE`; a whole document or batch is a transaction. Readers use explicit transactions, and snapshots use SQLite's [online backup API](https://www.sqlite.org/backup.html). SQLite allows multiple readers but only one concurrent writer. Lock timeout errors are surfaced rather than silently retried. See the [official transaction semantics](https://www.sqlite.org/lang_transaction.html).

Backup uses 128-page steps and a default 30-second cooperative deadline through the [Python backup progress callback](https://docs.python.org/3.12/library/sqlite3.html#sqlite3.Connection.backup). A callback deadline is not a hard interruption of disk I/O or SQLite lock waits. Timeout/failure leaves the destination for diagnosis; retry to a new path. Completion is followed by a full audit, which is outside that copy deadline.

Durability uses the runtime's existing journal mode and `synchronous=FULL` for writable connections. The process-exit control tests one interruption before commit. It does not certify disk flushes, network filesystems or physical power failure; SQLite documents these storage assumptions in its [atomic commit discussion](https://www.sqlite.org/atomiccommit.html). No distributed database or multi-node write service is claimed.

Unknown schema versions fail rather than being silently migrated. Future migrations must use a new explicit implementation and preserve historical identities. Hashes detect inconsistent changes; they are not signatures against an administrator who rewrites the complete database and records.

## Retrieval contract and ranking

`aim-retrieval-packet-v1` binds query, ranker, limits, library UUID, event sequence/hash, returned source metadata, offsets, excerpts, matched terms, integer scores and packet hash. Replaying a packet reruns ranking at its historical lifecycle snapshot and requires exact equality. Later versions are absent from that historical active set.

`unicode-token-coverage-v1` uses Python Unicode `\w+` after casefolding. For query token set Q and title-plus-chunk token set C, score is `|Q ∩ C|`. Return positive scores in descending order; break ties by source ID, start offset, then chunk ID. Apply a per-source cap before the global limit. Default limits are eight hits and two per source. This is a deterministic indexed baseline, not BM25 or embedding retrieval.

Original text is neither normalized nor casefolded for storage. Chunking uses character windows with a configurable overlap, not sentence segmentation. Chunks can split words or formulas; lexical matching misses paraphrases and morphological variants. Titles contribute to every chunk's score. Repeated mirrors may crowd results. These are known limitations requiring measured alternatives, not hidden quality claims.

Boundaries: 1 MiB per original, 64 MiB total originals, 5,000 stored versions including retired ones, 20,000 lifecycle events; chunk size 64–4,096 characters, overlap 0–half the chunk size; query up to 2,048 characters and 64 distinct word tokens; at most 32 hits and eight per source. Index, database and run snapshot bytes are additional to original-byte budgets. They have no separate quota in this version.

## Component boundaries

`DocumentResearcher.plan(question)` returns at most three unique queries. `propose(question,hits)` returns at most 32 unique retrieved chunk IDs and proposed excerpts. Inputs are copied. No source instruction can grant an action or execute a tool; the Controller only exposes local retrieval.

`ExcerptVerifier` independently checks exact proposed text, original bytes, offsets, quote hash and existing per-run Memory evidence. `DocumentJudge` chooses CITE or ABSTAIN for the exact-attribution target. The current contract permits only `probability=null` and `NOT_CALIBRATED`. A calibrated scientific forecast requires a separately defined event and training/evaluation record.

The Controller permits a citation only when both attribution PASS and Judge CITE occur. It imports unchanged originals into per-run Memory and binds the library chunk to the resulting Evidence. Review state remains separately versioned because prose excerpts do not fit numerical prediction contracts. This adds a domain adapter; it does not replace the project architecture or grant verifier authority to the Judge.

Markdown source material is fenced with a delimiter longer than any embedded backtick sequence. Titles, URIs and metadata are serialized as quoted data. The reference path does not execute document instructions or turn embedded labels into claim status.

## Reproduction and remaining work

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_library*.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_evidence_review.py' -v
.venv/bin/python -m aim.library_reproduce --regression --export reports/NEW_EVIDENCE_DIRECTORY
```

The [fixed protocol](experiments/phase-3f-protocol.md) records warm local exact-keyword retrieval, independent run reuse, lifecycle replay, two retained failed runs and one forced process exit. Exported JSON/Markdown logs are inspectable evidence; binary SQLite snapshots and source archives stay in retained runs and must be backed up separately. The exporter is intended only for this authored engineering study, not arbitrary private libraries.

Next: establish a reviewed document/question benchmark with source-level relevance labels, answerable/unanswerable cases, revisions, conflicting sources and frozen evaluation families. Compare coverage ranking with BM25 before an embedding or learned reranker experiment. Add explicit claim-to-source entailment assessment and contradiction annotations before synthesizing answers. PDF parsing, external literature connectors, signed provenance, authentication, erasure, shared-service storage and production-scale indexing remain unresolved build areas. Retrieval artifacts must pass a separate training-rights and split-contamination review before use as model data.

# Phase 3F — Shared evidence library and document review

**Implemented and locally audited, 2026-09-29 (Asia/Kolkata).** Run IDs use UTC and therefore begin with 20260928. This phase broadens Model-1 beyond synthetic numerical/symbolic inputs. It supplies a reusable document-memory foundation, not a completed general scientific model.

## What was built

| Component | Delivered behavior |
|---|---|
| Shared Library | Explicit creation; immutable UTF-8 originals, metadata, chunks and lexical postings in one local SQLite database |
| Intake | Hash-bound text/Markdown manifest, declared retrieval permission/privacy review, containment checks, atomic batches and idempotent reimport |
| Source lifecycle | New versions supersede active versions; withdrawal/retraction preserves history and excludes retired versions from new retrieval |
| Retrieval | Indexed Unicode-token coverage, deterministic ties, per-source/global budgets and hash-bound historical packets |
| Document Researcher | Separate bounded planning/proposal interface; reference backend extracts source excerpts |
| Document Judge | Separate CITE/ABSTAIN attribution policy; null confidence and explicit NOT_CALIBRATED status |
| Independent verifier | Rechecks proposed excerpt, source bytes, character offsets and per-run Memory evidence |
| Controller and state | Frozen library snapshot, copied component inputs, query/proposal limits, state revisions, citation gating and replayable final response |
| Operator commands | Initialize, ingest, list, inspect source/history, search, review, retire, audit and replay against current source statuses |
| Reliability | Serial write transactions, stable readers, exclusive snapshots, cooperative backup deadline, full content/index/history audit and retained failures |

The numerical/symbolic Controllers, learned backends and training objectives were not replaced. The prose adapter always labels source assertions **UNVERIFIED**. Attribution PASS is a scoped integrity result. A Judge cannot approve a fabricated quote into a final excerpt.

## Executed source and environment

Final study: `runs/20260928T205302-phase-3f-library-6b3f253a`.

- Commit: `9d88d1fcf74f84e64e7be29b8d4497c42df4e8f9`.
- Source/config/test/data hash: `9d8538aedaae4a2b0e889ed69dc9d8d7f603d13597339f4915ea36994f39ed31`.
- Python 3.12.14; SQLite 3.53.1; macOS 26.3.1 arm64; eight reported logical CPUs. Existing pinned neural dependencies were present.
- Final full regression: **260 tests passed, zero skips, 52.065 seconds**.
- Whole registered run: **57.261558 seconds**, including regression, benchmarks, lifecycle and failure controls; evidence export follows run completion.
- At run creation, Git reported the untracked operator guide and first evidence export. Implementation/config/tests were committed. Current source hashes were subsequently compared with the recorded snapshot: no differences.

Primary text evidence: [final study export](phase-3f-final-evidence/export.json), [test log](phase-3f-final-evidence/tests.log), [manifest](phase-3f-final-evidence/manifest.json), [benchmark records](phase-3f-final-evidence/benchmark.json). Full SQLite databases, per-run Memory, original documents and source archives remain in the retained local runs.

A subsequent [artifact audit](phase-3f-artifact-audit/audit.json) reconstructed all 240 saved benchmark packets, recomputed aggregates from recorded durations/ranks, re-audited all six actual-document reviews, checked exported file hashes and confirmed unchanged tested source. Its retained run is `runs/20260928T210015-phase-3f-artifact-audit-d33b7a43`. The [CSV](phase-3f-metrics.csv) contains the final benchmark table in machine-readable form.

The first completed study is also retained: `runs/20260928T204743-phase-3f-library-ecf3db0c`, source `d8d7fedf968f4d488d868b0b3f24b29acf06dda1`, 258 tests in 51.132 seconds, whole run 56.569271 seconds. Its [original export](phase-3f-evidence/export.json) is historical evidence before the later boundary fix. The fixed workload was not changed between studies.

## What actually passed

39 new tests cover the library, review loop and study runner. The remaining 221 regressions exercise existing model/training/research infrastructure, including local loopback distributed workers. No test was counted as passed by skipping a missing dependency.

- UTF-8/Unicode character offsets, original CRLF preservation, quote hashes and overlapping coverage.
- Idempotent intake, changed-version conflict rejection and all-or-nothing batch rollback.
- Four concurrent writers: one active version and three superseded versions, with a consistent event chain.
- A reader's transaction remains at its original snapshot while a writer prepares the next revision.
- Lock-timeout handling without partial source insertion.
- Immutable-row protections, original-text corruption, missing index postings and forged/rehashed retrieval packets.
- Exclusive and independent backups, retained timeout destinations and invalid deadline rejection.
- Historical packets replay after updates/retractions; reimport and latest-version retirement do not reactivate old versions.
- Two independent review runs produce equal final states against one shared snapshot.
- Fabricated quote rejection even with a Judge returning CITE; unretrieved proposal rejection; unsupported confidence declaration rejection.
- Original Memory, final response and state replay; source instructions remain quoted data.

These checks establish implementation behavior under the recorded environment. They do not establish linguistic entailment, scientific correctness, semantic search quality or robust calibration.

## Registered retrieval measurements

See the [protocol fixed before execution](../docs/experiments/phase-3f-protocol.md) and [configuration](../configs/library-study.json). Each authored note has a unique specimen keyword, eight repeated sections and no scientific result. Each size uses 16 exact-keyword queries, one warmup per query and five measured repetitions: **80 measured queries per size, 240 total**. The same 16 query identities repeat within each size; these are not 240 independent scientific tasks.

| Documents | Chunks | Original bytes | SQLite bytes | Ingest seconds | Full audit seconds | Median search ms | p95 search ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 128 | 55,520 | 1,286,144 | 0.013465 | 0.005454 | 0.436 | 0.459 |
| 128 | 512 | 222,080 | 5,001,216 | 0.113268 | 0.031163 | 1.351 | 1.583 |
| 512 | 2,048 | 888,320 | 19,910,656 | 1.584862 | 0.122380 | 5.533 | 5.736 |

At every size, document recall@5 and reciprocal rank@5 were **1.0**; the unmatched-vocabulary query returned zero hits. This deliberately easy unique-keyword workload checks ordering and inventory mechanics. It supplies no semantic relevance, paraphrase, scientific question-answering or generalization estimate.

Search timings include lifecycle replay and original checks for returned chunks. Packet verification occurs afterwards and is excluded. These are warm single-host measurements with no throughput concurrency experiment. No VIT machine was contacted or measured.

**Measured limitation:** at 512 documents the database is about 22.4 times the original text bytes. Explicit hash strings, indexes and postings have material overhead. Full-history replay also adds work proportional to lifecycle history; batch ingestion repeatedly validates history. This version is a bounded reference, not a web-scale search design. Compact index identifiers, quotas and state projections are candidate engineering optimizations that need new measurements and migration semantics.

## Actual AIM-document application

Final application: `runs/20260928T205434-phase-3f-project-documents-5cc31e18`, completed in **2.546845 seconds**, using the same final code hash. [Application evidence](phase-3f-project-evidence/summary.json) and [reproduction script](phase-3f-project-evidence/application.py).

The run copied the current versions of six existing AIM guides: TRAINING, JUDGE_SHIFT, HARDWARE_AND_CLUSTER, LANGUAGE_FITTING, PRETRAINING and DISTRIBUTED_PRETRAINING. Their original hashes, source versions and explicit local-use declarations are retained. This is project-authorized retrieval; no training permission or external-data license was inferred.

**58,461 original bytes, 69 chunks, six active source versions.** Fourteen actual CLI invocations completed: initialization, intake, inventory, history, source inspection, search, six review questions, live-library replay and full audit.

| Query | Attributed excerpts |
|---|---:|
| RLHF preference optimization RLVR reward calibrated Judge | 7 |
| VIT CPU cluster memory 7B DDP | 8 |
| checkpoint resume exact tokenizer BPE | 8 |
| calibration OOD uncertainty | 6 |
| free generation validation checker fitting | 8 |
| photonic superconducting neutrino | 0 |

All **37 selected excerpts** passed exact attribution and review replay. They remain **UNVERIFIED source assertions**. There are no independently reviewed relevance labels, so counts are not answer-quality scores. Active project documents can contain historical guidance; source activity alone does not establish that each sentence describes the latest project state. Temporal interpretation and contradiction handling remain future work.

The pre-fix application `runs/20260928T205113-phase-3f-project-documents-4e025dfb` also remains local. The published application evidence above comes from the final implementation.

## Failures, fixes and recovery

### Actual defect discovered after the first passing study

Creating an already existing, empty SQLite database with `user_version=99` incorrectly reinitialized it. The constructor's open-only path rejected the schema, but the create path did not. A new adversarial assertion reproduced the defect in **FAILED** run `runs/20260928T205211-phase-3f-adversarial-audit-892a3d42`. [Original failing log](phase-3f-adversarial-evidence/tests.log).

The fix requires schema version zero before initialization of an empty database. Both open and create paths now reject unknown versions. The failed run was preserved; the final 260-test regression includes the corrected assertion.

A review also identified unbounded backup retry risk. Backups now use incremental steps and a cooperative callback deadline, preserving the destination on failure. Deadline tests use a controlled clock; they are not evidence of a naturally occurring long-running backup or a hard I/O interruption guarantee.

### Registered expected failures

Each complete study retained two intentionally FAILED child runs: an intake source with the wrong SHA-256, and a Researcher proposing an unretrieved chunk. Their errors, source snapshots and status remain available. The final library still had zero sources after those rejected operations. [Final failed-run records](phase-3f-final-evidence/expected-failures.json).

### Process interruption

A separate child exited with code **23** after uncommitted source/index/event writes. Reopening the database produced exactly the pre-write empty inventory and history. [Recovery record](phase-3f-final-evidence/crash-recovery.json). This demonstrates one forced application-process exit before commit; it does not certify power-loss, disk failure or multi-node recovery.

## Decisions and limits

The [decision record D027](../docs/DECISIONS.md) keeps the approved component architecture and separate training objectives. One transactional local store simplifies content/index publication. Deterministic lexical ranking supplies an auditable reference. Frozen snapshots make old reviews reproducible while current-library replay exposes source retirement.

Remaining boundaries:

- No PDF/OCR/web ingestion, external literature connector, vector index or learned reranker.
- No prose synthesis, factuality/entailment verifier, independent-source reasoning or contradiction resolution.
- No human relevance/answerability study and no trained document Judge.
- No automatic schema migration, authenticated multi-user service, privacy erasure, distributed store or database-size quota.
- No new large model, pretraining run, RLHF dataset, combined RL objective or physical VIT result.
- All 2M allocation guards and conditional scale gates remain in force.

Canonical numerical and symbolic suite contents/checksums are unchanged. Source retirement is evidence lifecycle metadata, not an instruction to rewrite old experimental results or delete failed runs.

## Recommended next experiment

Build reviewed document/question cases with explicit source-level relevance, answerability, revisions, conflicting evidence and split provenance. Preserve procedural versus human label origins. Compare the existing coverage ranker with an independently checked BM25 implementation under matched chunk/query budgets before trying embeddings or neural reranking. Add claim-to-source assessment records and a contradiction workflow before generative synthesis.

The separate next ML study remains answer-field correctness and paired changed-evidence sensitivity on fresh diagnostic worlds. Large-scale training still requires representative reviewed data and actual sustained lab measurements.

## Reproduce and changed files

From `model-1/`:

```sh
.venv/bin/python -m aim.library_reproduce --regression --export reports/NEW_DIRECTORY
PYTHONPATH=. .venv/bin/python reports/phase-3f-project-evidence/application.py
```

The application script intentionally reads the current six guides and hashes them; to recreate an older input revision, use that revision's retained document copies or the recorded Git state. It does not silently overwrite existing runs. See the [operator guide](../docs/EVIDENCE_LIBRARY.md) for individual commands.

Implementation: `aim/library.py`, `aim/evidence_review.py`, `aim/library_cli.py`, `aim/library_reproduce.py`. Tests: `tests/test_library.py`, `tests/test_evidence_review.py`, `tests/test_library_reproduce.py`. Fixed config/protocol: `configs/library-study.json`, `docs/experiments/phase-3f-protocol.md`.

Documentation, handoff and all exported evidence paths/hashes are enumerated in [phase-3f-files.json](phase-3f-files.json), relative to the repository root. That inventory compares with phase-start commit `cc79a58` and excludes its own self-referential hash. [Machine-readable phase status](phase-3f-status.json) records the final result. Commits are grouped by implementation, validation, correction, evidence and documentation; line count is not an acceptance criterion.

# Phase 3F: persistent document evidence engineering

Registered before the full study on 2026-09-29. Configuration: `configs/library-study.json`.

## Questions and fixed scope

1. Can independent research runs reuse the same immutable source versions and reproduce their retrieved excerpts?
2. Can source replacement and retirement affect new retrieval without rewriting historical evidence?
3. Do explicit transaction boundaries prevent partial ingestion when a source conflicts, a writer times out, or a process exits before commit?
4. What local ingestion, audit and retrieval costs occur at 32, 128 and 512 authored documents?

These are engineering questions. This study does not test scientific reasoning, model quality, literature completeness, natural-language entailment or training-scale feasibility. The previous numerical and symbolic evaluations remain unchanged.

## Workload and measurement

Documents are deterministic authored notes with a unique `specimenNNNNN` keyword and eight repeated sections. They explicitly contain no scientific result. Store unchanged text in 512-character chunks with 64-character overlap. Use five returned chunks with a one-chunk-per-source cap.

At each size, ingest in transactions of at most 100 sources, audit the complete content/index/history, warm each of the first 16 exact-keyword queries once, then measure five repetitions of those queries. Record every query duration and returned source list, expected source, rank and packet hash. Report document recall@5, reciprocal rank, median duration and nearest-rank p95. Packet replay is executed outside the measured search duration. Search itself includes history validation and original-content checks for returned hits. Record SQLite version, source bytes, chunks and database bytes. These warm local measurements are not concurrency throughput or semantic-retrieval benchmarks.

Negative query: `absentvocabularytoken`, expected no matches. Its absence is a lexical control, not a calibrated abstention result.

## Lifecycle and failure controls

- Execute two identical evidence reviews against one shared library; require equal final states and independent run directories.
- Replace source version 1 with version 2; replay version 1's review and report SUPERSEDED as its current status.
- Retract version 2; new reviews must return no source from that URI. Version 1 must not be silently reactivated.
- Force a child process to exit with code 23 after source/index/event writes but before transaction commit. Reopen and require the previous audit to remain unchanged. This is an application process crash control, not power-loss certification.
- Retain separate FAILED runs for wrong intake hash and a Researcher proposing an unretrieved chunk.
- Unit/integration tests cover Unicode offsets, cross-library packets, forged quotes, a Judge attempting to approve a failed attribution, immutable records, index tampering, concurrent writers and reader snapshot stability.

## Acceptance and reporting

Require zero skipped tests in the full regression; exact lifecycle, replay and rollback assertions must pass. Record actual retrieval quality and timings regardless of outcome. Do not modify fixture sizes or query labels after inspecting results. Failures retain run manifests, source snapshots and errors. No promotion of neural architectures or parameter scale follows from this study.

Follow with a separately identified application run over existing AIM engineering documents. Any qualitative results from those documents are demonstrations, not labeled retrieval benchmark scores.

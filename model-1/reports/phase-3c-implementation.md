# Phase 3C — Operator audit and reviewed corpus build

Date: 2026-09-27. Status: **implemented and validated locally**.

## Built and connected

- Portable hardware observations with anonymous operator IDs, per-field measurement methods, explicit unknowns and content-bound import validation.
- A bounded retained filesystem write/fsync/readback probe with checks against a fresh local observation.
- Fleet trial preparation with stale/duplicate/runtime checks, exact policy-to-observation binding, per-node resource floors and explicit blockers; no automatic remote launch.
- Exact lexical near-duplicate and containment scanning, document/type/split coverage, quality flags and strict computation/output budgets.
- Complete per-document review records bound to corpus/audit/content hashes; blocked exports retained as failed attempts.
- Immutable reviewed subsets, original-byte/metadata preservation, full parent lineage and release replay.
- Eight new operator command groups, component tests and a complete review→release→distributed-training reproduction.

The [operator and corpus guide](../docs/OPERATOR_AND_CORPUS_READINESS.md) documents schemas, formulas, commands, predeclared acceptance gates and limitations. This extends the approved build; it does not alter the Researcher/Judge separation, training objectives or allocation limits.

## What actually worked

**192 tests passed, zero skips, in 69.774 seconds.** The full reproduction completed in **77.782 seconds**. It collected one real local host observation, wrote/read a retained 4 MiB probe, produced a blocked preparation plan, rejected a deliberately altered observation, audited the existing generated corpus, rejected an accept-all export, released the predefined subset and trained on it with two real local workers.

The 27 new tests include OS parser units, unsupported-platform behavior, hash/type validation, stale/future/duplicate records, missing permission/runtime/memory inputs, capacity floors, synthetic four-host planning, real local storage readback, foreign-record rejection, overlap enumeration against brute force, Unicode normalization, resource limits, corrupt objects, rehashed audit tampering, complete review membership/evidence requirements, failed export retention, byte/metadata preservation, release traversal/tampering and CLI behavior.

An earlier development run passed 24 tests in 4.150 seconds: `runs/20260927T103631-phase-3c-development-tests-9edd19a8`. Three tests and additional bounds/audit checks were added before the complete reproduction. No unexpected test or training failures occurred in these recorded runs. The two deliberate negative controls below remain FAILED; their rejection is expected evidence, not a reason to delete them.

## Measured local hardware and preparation status

The actual collector observed **Apple M3, arm64, eight physical/eight logical CPUs, 8,589,934,592 RAM bytes**, Python 3.12.14 and installed Torch 2.8.0. Available RAM remains null because the collector does not implement a macOS equivalent of Linux MemAvailable. The recorded disk-free value is a time-specific local observation, not a reserved allocation. No hostname, IP address or serial number was automatically included in the portable node record.

The 4,194,304-byte probe had exact byte readback and a matching retained hash. Write/flush/fsync took **0.001783 seconds**; cached read/compare took **0.000621 seconds**. These very short single-file timings are not sustained disk performance, shared-filesystem validation or proof of power-loss durability. The binary remains in the local run.

The actual plan is **BLOCKED: operator_policy_missing**. No lab permission was invented. Its workload has 6,496 parameters and an analytical FP32 parameter/gradient/Adam-moment floor of 103,936 bytes per rank, excluding activations/runtime. Available-memory validation still requires a supported measurement; adding a policy alone cannot establish training capacity.

Unit tests use explicitly synthetic records to exercise 1/2/4-host plan generation, differing runtimes, permission expiration and resource failures. They are not measured VIT hosts. The reported 70–84 i9/32GB/UHD770 machines remain unaccessed and unbenchmarked. Node identity is operator-assigned and hash-bound, not authenticated proof of distinct machines.

## Corpus audit and release outcomes

The input remains the 20-document, 3,707-byte authored integration fixture used in Phases 3A/3B. Repeated template families cross its train/validation/test splits; these were already documented limitations. This phase makes lexical flags executable rather than treating declared group IDs as sufficient protection.

Default audit parameters: NFC/casefold word/punctuation shingles of width five, minimum five shared shingles, Jaccard ≥ .8 or containment ≥ .9. These are provisional engineering thresholds, not tuned quality claims.

| Measurement | Original fixture | Released demonstration subset |
|---|---:|---:|
| Documents | 20 | 5 |
| Train / validation / test | 12 / 4 / 4 | 3 / 1 / 1 |
| Candidate pairs examined | 40 | 3 |
| Flagged pairs | 20 | 3 |
| Flagged cross-split pairs | 14 | 0 |
| Total per-document unique shingles | 750 | 181 |
| Posting-pair updates | 1,290 | 84 |

An accept-all review was rejected because 14 flagged cross-split pairs remained accepted. The passing demonstration follows the rule declared before reproduction: **retain train prose, validation math and test Unicode**. It accepts five documents and quarantines 15 in a new corpus. All original objects remain unchanged. The three within-training flags remain in the report and were explicitly accepted for this fixture; the released audit still says REVIEW_REQUIRED. It is not a claim that the subset is duplicate-free.

Release replay recomputed the audit/gate, checked all parent/audit/review/release hashes, and compared each accepted object's exact original bytes and metadata. Parent corpus:

`55d48cf01661c3396ddecbd34fecdfa4d929ff99f774ca65c6eb9bb4c462f02b`

Released corpus:

`ea55f6f59d1c31e396c0d68910d80ba8fd03142fd2a2490577a93095b68d1ba7`

The subset has deliberately different content types across splits and is not representative data or a useful generalization benchmark. Fixture review declarations are procedural construction evidence, not new human annotations or approval of an external corpus. The lexical scanner does not detect every form of template similarity, paraphrase, translation or short overlap; thresholds need independent assessment before production use.

## Actual training integration

The new released corpus entered the unchanged CPU/Gloo trainer. Two workers trained the 6,496-parameter random-init byte model for two optimizer updates using seed 17, batch two per rank, two accumulation microbatches and context 32. They consumed **510 scored targets**, agreed exactly on final model/optimizer replicas, and passed disjoint ownership of three training documents plus **four replayed rank updates** and final checkpoint cursor/count checks.

The job took **4.370 seconds** including launch/checkpoints/audits. Final fixed-prefix validation NLL was **5.538168 over 128 targets**. This confirms executable release-to-training integration. It does not establish useful research performance, data-mixture quality, sustained throughput or physical-node scaling. Uneven document ownership still repeats each rank's local partition independently.

## Failures and preserved records

Both negative-control children remain FAILED with full tracebacks and artifacts under the completed run:

- `invalid-node-import/20260927T104308-fleet-plan-5ce60d22`: node ID changed without updating its content hash; import rejected with `audit record hash mismatch`.
- `blocked-export/20260927T104308-corpus-reviewed-export-a8858697`: accept-all review retained flagged cross-split overlaps; `gate.json` records every blocker and no release was created.

Portable results include their status, relative path and failure-file SHA-256. Raw tracebacks remain local to avoid publishing runtime paths. The expected failures were not repaired in place or relabeled successful.

## Reproduction and evidence

Completed run: `runs/20260927T104155-phase-3c-build-11717358`.

Implementation source commit: `527b5093c9b8e95a66249dc706bda7f5b9996a8a`.

Source hash: `459af0b4069f75c0622bffc8e7acd192608c7075427326576e7d869fe114e3a3`.

- [Complete tests](phase-3c-evidence/tests.log)
- [Measured outcomes and retained failure records](phase-3c-evidence/results.json)
- [Local node observation](phase-3c-evidence/node.json), [storage probe](phase-3c-evidence/storage-probe.json), [blocked fleet plan](phase-3c-evidence/fleet-plan.json)
- [Original overlap audit](phase-3c-evidence/parent-overlap-audit.json), [review decisions](phase-3c-evidence/review-decisions.json), [released overlap audit](phase-3c-evidence/released-overlap-audit.json)
- [Release lineage](phase-3c-evidence/release.json), [release replay](phase-3c-evidence/release-audit.json)
- [Source/environment/artifact hashes](phase-3c-evidence/manifest.json)
- [Exact changed-file inventory](phase-3c-files.json)

Run from `model-1/`:

```sh
.venv/bin/python -m aim.readiness_reproduce --export reports/<new-directory>
```

The exporter replays release decisions, the blocked fleet plan, storage hashes, released overlap and actual training audits before copying portable evidence. References inside copied records resolve relative to their original retained run, not the portable export folder. Full input/source snapshots, original/released objects, probe binary, training checkpoints, rank logs and failure traces remain local. Canonical numerical/symbolic evaluation suites and historical reports/PDFs are unchanged.

## Decisions and next work

**Established:** the specified local observation, policy gating, lexical audit, review binding, immutable release and training integration paths work under the recorded tests. Missing policy and invalid/leaking inputs fail explicitly.

**Engineering decision:** retain these bounded preparation interfaces and make review evidence visible. Do not interpret a content hash as reviewer authentication, an OS memory figure as a job quota, lexical absence as semantic cleanliness or a reviewed subset as scale approval. Existing model/training defaults stay unchanged.

**Next implementation:** explicitly version tokenizer/vocabulary compatibility through supervised training and Researcher checkpoint adapters, starting with a from-scratch BPE pretraining→SFT bridge and exact continuation checks. Preserve byte compatibility and fail on ambiguous remapping. In parallel with that local build, obtain operator-collected lab observations and representative licensed corpus candidates for real capacity/data decisions.

The file inventory covers changes against baseline `23ef0d0`, excluding itself. This phase grows executable functionality and its evidence; source length and generated artifact size are not acceptance criteria.

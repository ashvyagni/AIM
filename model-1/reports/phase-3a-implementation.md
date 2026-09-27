# Phase 3A — Corpus, tokenizer and pretraining foundation

Date: 2026-09-27. Status: **implemented and validated at a miniature local scale**.

## Built

- Local corpus intake with source/version/hash metadata, explicit rights/privacy declarations, declared-group split checks, exact/normalized duplicate rejection and immutable content objects.
- A dependency-free byte tokenizer with unchanged legacy IDs/specification, plus a deterministic, from-scratch byte-pair candidate trained only on the train partition.
- Document-at-a-time packed next-token batches with explicit EOS/BOS/PAD behavior, token provenance, batch hashes and a restorable document/token/epoch/carry cursor.
- Native next-token pretraining with the existing causal decoder, separate from preference/RLVR/Judge objectives. Periodic checkpoints include optimizer, RNG, data/tokenizer identity and stream state.
- Exact continuation, a retained interruption/recovery drill, a compatible byte-pretraining→SFT bridge, CLI commands, configuration and complete reproduction tooling.

Read the [specification and commands](../docs/PRETRAINING.md). This is a working from-scratch training path. It is not a larger trained research model, and it does not replace the approved Researcher/Judge/Controller/Verifier/Memory architecture.

## What actually works

The generated corpus was ingested, both tokenizer variants trained or initialized locally, both native decoders completed next-token updates, and the byte checkpoint successfully initialized two SFT updates with a fresh optimizer and preserved parent hash. BPE checkpoints are explicitly rejected by the existing byte-only SFT path; no vocabulary remapping occurs silently.

For each tokenizer, an uninterrupted 12-step run matched a 6-step run continued to step 12. All eleven recorded comparison fields passed: model tensors, optimizer, Torch RNG, step, scored-token count, cursor, corpus hash, tokenizer, stable configuration, validation metrics and continued batch hashes.

The separate application-interruption drill saved step 4, deliberately raised an exception, retained a **FAILED** run, then resumed to step 12. Recovery matched the uninterrupted byte run on the same eleven checks. The failure was not reclassified as a successful run. This is an application-level exception test; it does not establish resilience to power loss, OS kills, device failure or distributed-node loss.

## Data and tokenizer evidence

The fixture contains **20 project-generated documents, 3,707 UTF-8 bytes**: 12 train, four validation and four test. It includes prose, code-as-text, elementary math and Unicode. It contains no imported passages or real-person records. Rights/privacy records describe its construction; they are not a completed review of any external corpus. The repeated template families cross splits, so this is not a family-generalization benchmark.

Corpus hash: `55d48cf01661c3396ddecbd34fecdfa4d929ff99f774ca65c6eb9bb4c462f02b`.

| Validation encoding | Vocabulary including special IDs | Content tokens | Bytes per token | Exact round trips |
|---|---:|---:|---:|---:|
| UTF-8 bytes | 259 | 744 | 1.00000 | 4/4 |
| 32-merge byte-pair candidate | 291 | 538 | 1.38290 | 4/4 |

The BPE candidate uses 27.69% fewer content tokens on this fixture. This is a compression result only; no production tokenizer is selected. The fitter is deliberately capped at 65,536 training bytes and 128 merges. Existing byte-token model/checkpoint contracts remain compatible.

## Training observations

Both runs used seed 17, 12 updates, batch 2, context 64, two layers, width 32 and CPU FP32. Vocabulary size changes the parameter count. These are engineering allocations, not final AIM model sizes.

| Variant | Parameters | Scored training targets | Initial validation token NLL | Final validation token NLL |
|---|---:|---:|---:|---:|
| Byte | 26,880 | 1,528 | 5.585034 | 5.045081 |
| Byte-pair | 27,904 | 1,525 | 5.684565 | 5.322551 |

Validation uses a fixed four-batch prefix: 510 scored byte targets and 509 BPE targets. **Do not compare NLL directly between tokenizers.** The prefixes cover different byte positions, vocabularies differ, and this is neither an equal-parameter nor a matched-byte study. Lower loss after a few updates on related constructed text does not demonstrate mathematical or research capability.

Packed windows allow causal attention across document boundaries; BOS prediction and PAD positions are masked, EOS is trained. The stream is deterministic and sequential, without shuffled mixtures or distributed sharding. These choices are explicit reference behavior, not final production decisions.

## Tests, failures and timing

**154 tests passed, zero skips, in 27.096 seconds.** Fourteen new tests cover permissions/review declarations, hashes, duplicate/group leakage, traversal/symlinks, corrupt objects/indexes, deterministic train-only tokenization, Unicode round trips, finite packing without lost transitions, cursor continuation across epochs, byte/BPE training resume, interruption recovery, forbidden test-text access, SFT compatibility and portable traceback redaction. Earlier numerical/symbolic tests passed too; canonical suite files were not changed.

The full final build run took **29.442 seconds** on the recorded local macOS arm64 host with eight logical CPUs. Torch training used one CPU thread. The byte/BPE full child runs recorded 0.519/0.088 seconds, of which 0.036/0.030 seconds were update-loop time. These intervals are too short for reliable throughput planning; they are smoke-test timings, not VIT fleet measurements. No VIT machine, physical network or GPU was accessed.

The first targeted development run passed 13 tests in 5.743 seconds. An initial complete build also passed 153 tests. Its exported injected-failure traceback still contained local runtime paths. Export formatting was corrected and covered by an additional test; the full build was repeated. The first completed run remains at `runs/20260926T184232-phase-3a-build-de1bfa86`, with its original review export retained locally under `private-review-export/`. No numerical result was discarded or retuned. No unexpected training or infrastructure failure occurred; the recorded training failure is the deliberate recovery drill.

## Reproduction and artifacts

Final completed run:

`runs/20260927T051600-phase-3a-build-b9b61ea2`

Source commit: `5fea7ac932d6bb8b1a9fabf2221e48b965b2c7ad`.

Source bundle hash: `2ad86ba02adda0068738864fcd4fa32f886239a06e00df6bfb6c12da796ca640`.

Retained failed child: `interruption-drill/20260927T051629-pretrain-664ec44c` within that run. Its step-4 checkpoint, sidecar, configuration, events and traceback remain intact.

- [Complete test log](phase-3a-evidence/tests.log)
- [Training, continuation, recovery and SFT records](phase-3a-evidence/results.json)
- [Tokenizer comparison](phase-3a-evidence/tokenizer-comparison.json)
- [Constructed fixture text and hashes](phase-3a-evidence/fixture.json)
- [Retained injected-failure traceback, with portable paths](phase-3a-evidence/injected-failure.txt)
- [Configuration, environment, source and export hashes](phase-3a-evidence/manifest.json)
- [Exact changed-file inventory](phase-3a-files.json)

Run from `model-1/`:

```sh
.venv/bin/python -m aim.pretrain_reproduce --export reports/<new-directory>
```

Full source snapshots, input manifests, content-addressed corpus, periodic/final checkpoints and optimizer/RNG state remain local. Portable reports include constructed fixture text only. The original research handbook and earlier result bundles are unchanged.

## Decisions and next implementation

**Established by this run:** local intake, tokenization, next-token training, exact CPU continuation, the defined recovery drill and byte-token SFT initialization work on the fixture.

**Engineering decision:** retain the byte baseline and BPE experiment separately; keep allocation guards. Do not infer permissions, silently repair corpus records, reinterpret compression as model quality, or promote this tiny model.

**Unresolved:** representative licensed corpus acquisition, semantic deduplication, large-scale tokenizer fitting, data mixtures/shuffling, document-isolated attention, BPE-aware supervised training, scientific evaluation and physical hardware throughput.

**Next build:** deterministic shard/rank partitioning and data-parallel cursor/checkpoint recovery at small local scale. Define rank/world-size ownership and failure semantics; test disjoint coverage, equal batch counts and exact restart before physical-node deployment. Actual 100M–300M proxy training remains conditional on corpus review and measured hardware; ~1B/7B+ remain later decisions.

The [inventory](phase-3a-files.json) records every changed phase file against baseline `5527828`, excluding itself. Changes include data/training/tokenizer modules, CLI, tests, configuration, evidence, documentation and handoff.

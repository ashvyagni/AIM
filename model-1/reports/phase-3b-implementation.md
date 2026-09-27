# Phase 3B — Distributed corpus training and recovery

Date: 2026-09-27. Status: **implemented and validated on local CPU workers**.

## What was built

- Deterministic document ownership per rank, with complete/disjoint partition audits and rank-bound resumable token streams.
- Actual CPU/Gloo DDP next-token training of AIM's native random-init decoder, with microbatch accumulation and one global valid-token denominator.
- Worker agreement on code, configuration, Python/Torch, corpus, partition and tokenizer before training.
- Coordinated checkpoint bundles: one common model/optimizer record, per-rank RNG/cursor/counter records, member hashes and a publication manifest. Pending checkpoints cannot load.
- A bounded local supervisor for 1/2/4 workers; preflight, inspection, initialization-export commands; failure logs and cleanup of its own launched process group.
- Read-only replay of consumed batches, masks, cursors and counts against the final checkpoint.
- An independent serial trainer, exact continuation comparisons, two publication-boundary failure drills, a four-worker smoke job and compatible SFT initialization.

The [specification](../docs/DISTRIBUTED_PRETRAINING.md) documents contracts, mathematics, commands, acceptance gates and limits. The Researcher/separate Judge/Controller/Verifier/Memory architecture and separated post-training objectives are preserved.

## Actual execution

The complete reproduction launched **11 distributed jobs: nine COMPLETED and two deliberately FAILED**, in addition to distributed jobs inside the unit suite. It also ran two independent serial reference trainings and a two-update SFT initialization check. All jobs used loopback on one macOS arm64 host reporting eight logical CPUs. Training used FP32 and one Torch thread per process, Python 3.12.14 and Torch 2.8.0.

The corpus is the same 20-document, 3,707-byte project-generated integration fixture: 12 train, four validation and four test. Corpus identity:

`55d48cf01661c3396ddecbd34fecdfa4d929ff99f774ca65c6eb9bb4c462f02b`

Training does not read test text. Repeated template families cross fixture splits, so these are systems checks rather than family-generalization evidence. No external literature corpus, pretrained weights or VIT hardware was used.

Full byte/BPE runs used seed 17, six optimizer updates, two ranks, batch two per rank, two accumulation microbatches, context 32, width 16, one layer, two query heads, one KV head and FFN width 32. Checkpoints were published every two steps. BPE used 32 train-only merges. Both model sizes remain far below the unchanged 2M allocation guard.

| Full-run measurement | Byte baseline | BPE experiment |
|---|---:|---:|
| Parameters | 6,496 | 7,008 |
| Vocabulary | 259 | 291 |
| Scored training targets across ranks | 1,528 | 1,526 |
| Initial fixed-prefix validation token NLL | 5.530621 | 5.677408 |
| Final fixed-prefix validation token NLL | 5.438027 | 5.607987 |
| Serial-reference maximum absolute parameter difference | 3.678724e-8 | 1.192093e-7 |
| Predeclared serial comparison tolerance | 2e-6 | 2e-6 |
| Exact step-2 → step-6 continuation | PASS | PASS |
| Final model/optimizer replicas agree | PASS | PASS |
| Replayed rank updates | 12 | 12 |
| Complete parent-job elapsed seconds | 5.027 | 7.936 |

Validation scores 128 targets from a fixed two-batch prefix. Vocabularies and covered byte positions differ: **do not compare token NLL as cross-tokenizer model quality**. Neither training loss nor these micro-models establish research capability. The serial oracle consumes the identical rank microbatches, differentiates the global token mean without DDP, and compares final parameters and cursors; it is numerical agreement within tolerance, not bitwise equivalence.

The separate four-worker job completed two updates, consumed 1,020 scored targets globally, and passed complete/disjoint ownership of 12 training documents, eight replayed rank updates and exact replica agreement. Its parent job took 4.593 seconds. Four processes shared one physical host; this is not a four-node scaling result.

## Exact continuation and checkpoint failure results

For byte and BPE, a two-step run continued to six steps matched uninterrupted training on **23 recorded checks**: nine common fields and seven fields for each of two ranks. These include model tensors, optimizer, model/config/data/tokenizer identities, step, global/local counts, rank identity, stream cursor and Torch RNG. Parent bundle hashes legitimately differ because lineage differs; they are not treated as numerical state.

| Injected worker exception at step 4 | Retained checkpoint state | Recovery start | Final comparison with uninterrupted byte run |
|---|---|---:|---|
| Before publication | Step 2 VALID; pending step 4 INCOMPLETE/unloadable | Step 2 | All 23 checks exact |
| After publication | Steps 2 and 4 VALID | Step 4 | All 23 checks exact |

The two parent jobs remain **FAILED**. Their launcher logs contain the explicit injected-failure marker; export records include each log's SHA-256. No failed experiment or partial checkpoint was deleted. They test a worker exception, peer/launcher propagation and restart from completed optimizer steps. They do not test OS kills, physical node loss, power loss or restart during a microbatch. A peer terminated by the launcher can lack a final per-rank status; the parent failure and launcher log remain recorded.

The byte bundle exported an initialization-only artifact and completed two compatible SFT updates with a fresh optimizer. Its SFT response NLL moved from 5.557933 to 5.546492, while preference accuracy fell from .53125 to .5. These are bridge diagnostics, not a preference improvement claim. BPE→SFT remains unsupported by the current byte-only supervised path.

## Tests, timing and failures

**165 tests passed, zero skips, in 66.740 seconds.** The 11 new tests cover partition completeness and isolation, cursor binding, partial/corrupt/missing/disagreeing checkpoint members, manifest path tampering, global-token gradient normalization, configuration guards, real two-worker continuation, serial agreement, replay tampering, failure recovery, BPE and world-size-change rejection. Earlier numerical/symbolic/objective tests passed; canonical evaluation files and their sidecars are unchanged from baseline `a66bcbd`.

The full reproduction took **125.536 seconds**, including the complete regression suite, all training jobs, failure drills and audits. Individual parent-job times include worker launch, rendezvous, checkpoint writes and audits. The runs are too short and small for sustained throughput estimates or comparisons across worker counts. No fleet speedup, physical network bandwidth or larger model feasibility is established.

An earlier targeted development run passed 11 tests in 28.409 seconds at `runs/20260927T055108-phase-3b-development-tests-d7e63f2a`. Final audit hardening subsequently bound per-rank totals/cursors to checkpoint records and added its negative assertion; the full reproduction above includes that version. There were no unexpected training/test failures in these two recorded runs. Deliberate checkpoint-failure tests and the two full reproduction drills are retained as expected negative cases.

## Reproduce and inspect

Completed run: `runs/20260927T060039-phase-3b-build-7087fd85`.

Implementation source commit: `bd01edd20f484b259773f6a86bde3c36a76589f6`.

Source bundle hash: `f7c23e1d7e16605cc839eba37b7577dcfc3df9e29b3afaf822fdb09647aed4c1`.

Retained failed children, relative to that run:

- `jobs/20260927T060226-distributed-pretrain-launch-d4e4da30` — before publication.
- `jobs/20260927T060232-distributed-pretrain-launch-b5524430` — after publication.

Portable artifacts:

- [Complete regression log](phase-3b-evidence/tests.log)
- [Training, replay, serial, continuation, failure and SFT results](phase-3b-evidence/results.json)
- [Byte specification](phase-3b-evidence/byte-tokenizer.json) and [BPE specification/provenance](phase-3b-evidence/bpe-tokenizer.json)
- [Configuration, environment, source and artifact hashes](phase-3b-evidence/manifest.json)
- [Exact changed-file inventory](phase-3b-files.json)

From `model-1/`:

```sh
.venv/bin/python -m aim.distributed_reproduce --export reports/<new-directory>
```

The exporter re-audits successful jobs, exact continuation, serial comparisons, failure log hashes and exact recovery before publication. Complete source snapshots, fixture objects, per-rank manifests, checkpoints, launch commands, logs and failed directories remain local under the completed run. The Git evidence bundle excludes checkpoint binaries and raw launcher logs containing local paths. Back up the local run separately; a fresh clone can reproduce new evidence using the command above.

## Decisions, unresolved work and next experiment

**Established by this run:** the specified local fixed-topology data-parallel path performs real updates, preserves data ownership, publishes complete checkpoints, resumes exactly in the recorded environment and agrees with the independent serial calculation within the declared tolerance.

**Engineering decision:** retain this implementation as the systems reference and keep all large-allocation gates. DDP replicates the complete weights/optimizer on each worker. A common checkpoint file reduces saved duplication; it does not pool host RAM. No final parameter count, production tokenizer, calibrated language Judge or blended reward is selected.

**Limitations:** static rank partitions repeat independently, potentially overexposing shorter partitions; no shuffled mixture or load balancing. No optimizer/model sharding, elastic membership, KV cache or multi-host scheduler. Publication uses directory rename without a power-loss/fsync guarantee. Trusted shared storage and peers are assumed. Source/runtime agreement is checked between current workers; exact portability across changed libraries/devices is not established. The local port selection has a release/bind race and retains launch errors rather than hiding them.

**Recommended next build:** portable lab-operator audit collection with measured versus reported fields and import checks, plus representative corpus review and near-duplicate controls. Prepare bounded physical-host trials after the operator supplies actual access/environment/storage details. Measure one host, then matched 2/4-host workloads and real checkpoint recovery before proposing scale. Corpus readiness and physical throughput still gate 100M–300M proxies, followed by conditional ~1B and 7B+ work.

The [inventory](phase-3b-files.json) hashes every changed phase file against baseline `a66bcbd`, excluding itself. It covers source, tests, configuration, protocol, result artifacts, README updates, decision/specification updates and the owner handoff. Historical PDFs and earlier evidence bundles remain unchanged.

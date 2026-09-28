# AIM Model-1 — implementation directive and handoff

**For the next implementation agent or ML engineer.** Updated 2026-09-29. Work in the existing `AIM/` workspace; the canonical code directory is `model-1/`. Continue the approved architecture and current implementation. This directive supplies a concrete starting point, not a request to redesign AIM.

## Current owner direction — Continue after the completed Phase 3F document-memory build

Read the [Phase 3F report](../model-1/reports/phase-3f-implementation.md), [library/review guide](../model-1/docs/EVIDENCE_LIBRARY.md) and [registered protocol](../model-1/docs/experiments/phase-3f-protocol.md). All **260 tests passed, zero skips**. Shared local UTF-8 intake, exact chunks, indexed retrieval, source versions/retirement, historical replay, snapshot reviews and separate Researcher/Judge/verifier interfaces now work at the documented engineering scope.

**Latest registered run:** `model-1/runs/20260928T205302-phase-3f-library-6b3f253a`, 57.261558 seconds including 52.065 seconds of regression. Source `9d88d1fcf74f84e64e7be29b8d4497c42df4e8f9`, code hash `9d8538aedaae4a2b0e889ed69dc9d8d7f603d13597339f4915ea36994f39ed31`. The fixed 32/128/512-document workload recorded 240 repeated exact-keyword queries. Final 512-document median was 5.533 ms; these are local lexical fixtures, not semantic-quality or physical-cluster results.

**Application:** `model-1/runs/20260928T205434-phase-3f-project-documents-5cc31e18` executed 14 CLI commands over six actual AIM guides: 58,461 original bytes, 69 chunks, six reviews and 37 attributed excerpts. All assertions remain UNVERIFIED. One actual unknown-schema initialization failure was retained and fixed; two registered rejection failures and the precommit process-exit control remain recorded. Full source snapshots/databases stay local, with portable evidence under `model-1/reports/phase-3f-*-evidence/`.

**Next engineering action:** establish a versioned reviewed relevance/answerability corpus with source versions, conflicting statements and family-disjoint splits. Add explicit claim-to-source assessments and contradiction records before generative synthesis. Compare the current coverage ranker with BM25 under matched budgets before embeddings/learned retrieval. Preserve label origin: authored procedural cases are not human-reviewed evaluation. No final model/Judge scale or combined reward is promoted.

**Separate next ML experiment:** answer-field correctness and paired evidence-change diagnostics remain pending. The Phase 3E learned capability limitations below still apply. No lab hosts or representative training corpus have been validated; do not remove allocation guards or infer physical training capacity from the library benchmark.

## Earlier completed fitting build — Phase 3E

Read the [Phase 3E implementation report](../model-1/reports/phase-3e-implementation.md), [diagnostic guide](../model-1/docs/LANGUAGE_FITTING.md) and [registered study](../model-1/docs/experiments/phase-3e-protocol.md). All **221 tests passed, zero skips**. Eight byte/BPE SFT arms completed 32 checkpoint/baseline measurements, 256 probe measurements and 40 real Controller replays. The standalone diagnostic CLI exactly reproduced its selected registered measurement under the stated one-thread environment.

**Latest completed run:** `model-1/runs/20260928T103031-phase-3e-language-fitting-01036454`, 798.968 seconds including 459.576 seconds of regression tests. Executed source `a1748ea43d0c259c92d6d82988f68828c874c252`, code hash `81e052ac707dd35519be9a399603d945573fde81e0fa018023be187941f9e027`. Portable evidence is in `model-1/reports/phase-3e-evidence/`; full checkpoints and ledgers remain local. All declared arms and budgets are retained.

**Outcome:** output structure improved after 384 updates, but every arm had zero validation checker passes. All final symbolic probes met the output contract and failed the exact checker. Two of 32 neural Controller episodes produced verified claims, both training cases from numerical byte/seed-29; no neural validation case did. All eight reference episodes verified claims. These are a few repeated, exposed fitting worlds, not independent research benchmarks or proof of generalization. No default model, tokenizer or scale is promoted.

**Historical next engineering action, now completed in Phase 3F:** bounded local document/chunk ingestion, exact original spans, cross-run retrieval and source update/retraction records. Retrieval support must not imply factual verification.

**Separate next ML experiment:** register answer-field and paired evidence-change diagnostics to distinguish template fitting from prompt-sensitive correct content. Larger parameters, loss reweighting and constrained decoding remain hypotheses. Exclude all previously exposed worlds from any fresh capability holdout. Do not rerun the completed fitting study merely to begin the next build.

## Earlier completed audit — Phase 3D

The owner's continuation was used to execute the previously deferred audit. Read the [executed audit report](../model-1/reports/phase-3d-audit.md) and [tokenizer-compatible post-training specification](../model-1/docs/TOKENIZED_POSTTRAINING.md). All **208 tests passed, zero skips**, after four adversarial failures were reproduced and fixed. The earlier [build-only report](../model-1/reports/phase-3d-build.md) remains an accurate historical record, not current validation status.

Exact continuation passed for all six byte/BPE SFT/preference/RLVR combinations and two retained interruption-recovery drills. A two-worker BPE initialization export entered SFT. Twelve actual Controller episodes replayed: four reference episodes each produced a verified claim; all eight neural episodes exhausted the token budget with invalid JSON and zero hypotheses. The default architecture, separate Judge and legacy training remain. No learned backend or larger scale is promoted.

**Phase 3D completed run:** `model-1/runs/20260927T160636-phase-3d-audit-3892fe0c`, 59.249 seconds, exported to `model-1/reports/phase-3d-audit-evidence/`. Audited source `2ef34945f657f35af0d4b770104936ad8581e98d`, source hash `b00ec44062651c90cad8fb4e74fb85c7ee5150a63c8b2a2b93e3dcd9b9a66a8f`. The separate adversarial run and both injected interruptions remain FAILED. Their correction/recovery does not erase that history.

Its proposed fitting-diagnostics follow-up is completed in Phase 3E above. Phase 3D's Controller cases remain exposed integration fixtures. Preserve the original numerical/symbolic canonical suites and all retained failures.

## Earlier completed validation — Phase 3C

**Historical state:** Phase 3C is implemented and validated. Read its [report](../model-1/reports/phase-3c-implementation.md) and [operator/corpus specification](../model-1/docs/OPERATOR_AND_CORPUS_READINESS.md). All 192 tests passed, zero skips. Local hardware observations, bounded storage probing, policy-bound fleet preparation, lexical contamination audits, complete document reviews and immutable subset release/replay work. A reviewed fixture subset entered actual two-worker training and passed data/checkpoint audits. Phase 3D subsequently added tokenizer-compatible post-training; actual lab/data validation remains pending.

**Phase 3C completed run:** `model-1/runs/20260927T104155-phase-3c-build-11717358`, 77.782 seconds, exported to `model-1/reports/phase-3c-evidence/`. Source commit `527b5093c9b8e95a66249dc706bda7f5b9996a8a`. The tampered-node import and overlapping accept-all export remain FAILED. The actual fleet plan is BLOCKED without an operator policy. The passing engineering subset retained five of 20 documents, preserving all parent objects, and trained for two updates/510 scored targets on two local workers. Do not relabel this as external-data approval or VIT deployment.

**Phase 3B completed run:** `model-1/runs/20260927T060039-phase-3b-build-7087fd85`, 125.536 seconds, exported to `model-1/reports/phase-3b-evidence/`. Nine distributed jobs completed; two deliberately failed before/after publication and remain FAILED with logs/checkpoints intact. Both recoveries matched uninterrupted training on all 23 recorded common/rank state checks. These are worker-exception drills on one host; physical-node deployment and power-loss recovery are untested. The implementation source commit is `bd01edd20f484b259773f6a86bde3c36a76589f6`. Do not restart this completed phase as unfinished work.

**Completed run:** `model-1/runs/20260927T051600-phase-3a-build-b9b61ea2` completed in 29.442 seconds and exported to `model-1/reports/phase-3a-evidence/`. Its byte/BPE continuation and application-interruption recovery passed all eleven state checks. The failed child is deliberately retained. The earlier completed Phase 3A run is preserved too; its report-export runtime-path issue was corrected and regression-tested before the final full rerun. Do not restart these runs as unfinished work.

Phase 2C's `model-1/runs/20260926T181327-phase-2c-build-5e038999` is complete. Its formula reference verified eight expansions; learned Researchers verified none. The RLVR+Judge arm failed generation before reaching judgement; learned-Judge integration was exercised with the formula proposer. Keep those learned models experimental.

Phase 2B.1's `model-1/runs/20260925T193029-paid-evidence-reproduction-751ae7ad` and its audited 1,248 episodes are complete too. Its paid-evidence gate failed, and selective acquisition bought everywhere. All historical runs/checkpoints remain local; portable evidence is versioned in Git. Earlier reports retain their historical results.

## Read before changing code

1. The complete [research handbook](AIM_Complete_Research_Architecture_and_Project_Handbook.pdf) and relevant [final-design specifications](final-design/).
2. [Model-1 README](../model-1/README.md), [decision record](../model-1/docs/DECISIONS.md), [architecture](../model-1/docs/ARCHITECTURE.md), [training specification](../model-1/docs/TRAINING.md), [evaluation specification](../model-1/docs/EVALUATION.md) and [hardware protocol](../model-1/docs/HARDWARE_AND_CLUSTER.md).
3. The [actual phase report](../model-1/reports/phase-1-implementation.md), portable evidence bundle, and existing implementation/tests/configs. Verify artifact paths before citing results.

Historical documents mix established evidence, recommendations, hypotheses, open questions and hardware-dependent decisions. Preserve those distinctions. In particular, later explicit owner requirements supersede the older suggestion to borrow pretrained weights: **AIM's learned models start from random initialization.**

## Architecture to maintain

Researcher + separate Judge + deterministic Controller + independent Verifiers + provenance-aware Memory. Confidence is a forecast of a defined event. Only an actual scoped verifier pass permits a corresponding VERIFIED claim. Store failed and unresolved hypotheses and evidence; do not let component agreement substitute for verification.

Keep SFT, preference learning, RLVR and Judge calibration independently trainable and measurable. The current progression is SFT → DPO preference baseline → verifier-grounded RLVR → separate calibrated Judge → future integrated loop training. Do not implement a blended reward without a mathematical specification, baselines, measured results and appropriate review.

## What already exists

- A working numerical reference investigation from question to provenance-linked final response.
- A native dense decoder trained from random weights, UTF-8 byte tokenizer and checkpoint lineage/resume.
- Actual SFT, DPO, finite-candidate REINFORCE and separate proper-score Judge updates.
- A strict neural Researcher adapter; phase-1 arithmetic weights failed hypothesis generation, while later research-specific weights learn valid structure but still fail capability gates.
- Checkpoint-versioned plain/worked hypothesis formats, fresh world partitions, frozen multi-seed comparisons, independent process diagnostics and tested Researcher continuation.
- Staged arithmetic tasks, fitting/transfer diagnostics, exact task accounting, a scoped observation verifier and read-only memory audit.
- Separate five/seven-feature Judges, grouped shift fixtures, calibration freeze, complete forecast/policy metrics and a strict opt-in Controller adapter.
- Optional paid acquisition, immutable purchased observations, shared target decisions, question reward capped at one, and an independent temporal/accounting audit.
- A separate symbolic research loop, bounded exact rational verifier, strict native Researcher adapter, symbolic Judge and independently runnable SFT/preference/RLVR/calibration stages.
- Corpus intake with declared rights/privacy and split checks, byte/BPE tokenizer comparison, document-at-a-time next-token batches, full stream cursors, periodic pretraining checkpoints and a byte-checkpoint SFT bridge.
- Corpus-driven CPU/Gloo workers, global-token gradient accumulation, fixed document ownership, per-rank cursors/RNG, coordinated checkpoint bundles, local supervisor, preflight/inspection/export commands, replay audits and independent serial reference.
- Anonymous-ID local observations, explicit unknowns, bounded retained storage probes, policy/observation bindings and fixed-workload fleet preparation without remote dispatch.
- Exact lexical overlap/containment reports, complete hash-bound document reviews, blocked export records, immutable subsets and original-byte/metadata release replay.
- Versioned state/claims/evidence, deterministic action budgets, bounded subprocess tools, exact numerical/provenance verifiers and append-only event replay.
- Shared versioned local document library, atomic intake, exact source chunks, indexed cross-run retrieval, retirement/history, snapshot evidence reviews and original-attribution replay.
- Regression tests, public evaluation fixtures, complete reproduction command, local CPU and Gloo/DDP benchmark infrastructure.

The 90,624-parameter initial decoder, 228,096-parameter structured Researcher and 225/289-parameter feature Judges are micro-scale mechanism checks. They are not final model/Judge size choices and do not replace the 100M–300M proxy → ~1B systems → conditional 7B+ roadmap. Do not represent the current system as a general research AI.

## First actions

```sh
cd model-1
.venv/bin/python -m unittest discover -s tests -v
```

Use the documented environment setup if the local `.venv` is missing. Read the latest report and existing bundle's `tests.log`, `results.json`, case records and export manifest before launching experiments. A missing torch environment is not a successful neural test result. The documented reproduction command can repeat a study if needed; do not rerun a completed study merely to begin the next phase.

Continue the [implementation plan](../model-1/docs/IMPLEMENTATION_PLAN.md) with evidence quality and explicit claim assessment after the completed Phase 3F library build. Preserve library snapshots, lifecycle semantics and attribution/factuality separation, alongside exact tokenizer identities, separately encoded loss boundaries, frozen references, stage-specific resumes and strict model adapters. No lab hosts or access details have been supplied; do not invent deployment results. Actual 100M–300M proxy training remains gated on representative data and physical hardware evidence.

Phase 3C observed the local Apple M3 host with 8 GiB RAM; it did not measure any VIT desktop. Available RAM on macOS stays unknown in the current collector. Four-host planning records exist only as synthetic unit-test fixtures. Policies/reviews are unsigned declarations tied to hashes, not authenticated authority or legal certification. The corpus audit found 14 flagged cross-split pairs at its provisional lexical thresholds. The five-document export keeps train prose, validation math and test Unicode; its three within-train flagged pairs remain visible. This demonstrates review/export mechanics, not production data quality or a recommended mixture. Do not use line count or generated evidence volume as an acceptance metric.

Phase 3B retains the 2M allocation guard. DDP replicates model/optimizer state; the common checkpoint file only reduces artifact duplication. Rank ownership is static, with independent local epochs; shorter partitions repeat sooner. World-size changes fail until a resharding protocol is specified. Only published step directories resume; pending directories remain for diagnosis. Shared-filesystem visibility, process supervision across real hosts, fsync/power-loss behavior and heterogeneous-node numerical reproducibility remain open. The local launcher deliberately uses loopback and must not be reused as a remote deployment tool.

Phase 3A's 20-document, 3,707-byte corpus is project-generated integration text with repeated template families across splits. It does not complete an external corpus review. The 32-merge BPE candidate compresses the four validation documents but is not a selected production tokenizer. Token NLL is not comparable across vocabularies. Cross-document causal attention is currently allowed, and BOS/PAD prediction targets are masked. Existing SFT accepts byte-compatible pretraining weights only; BPE transfer must be explicitly implemented rather than silently remapped. The retained fault drill injects an application exception, not an OS kill or multi-node failure.

Phase 2C's first symbolic slice is complete. Preserve its numerical/symbolic claim distinction and canonical suites. A stronger task distribution, independent second checker and language-capable Judge remain research work. The symbolic smoke run's preference labels were programmatic; it did not collect human feedback. Do not interpret higher finite-candidate RLVR reward as successful free generation or treat the weak syntactic Judge as a final mathematical reasoning model.

Phase 2B.1's acquisition/target costs are explicit illustrative units; failures are paid and stop unresolved. The shared forecast remains a heuristic and the selective band learned no information-value policy. Future acquisition learning needs fresh data, expected-value targets, coherent question probabilities, variable probe choices and explicit recovery semantics. All previous holdouts are exposed. Do not tune them or enlarge the model merely to make a gate pass.

Phase 2A.2 found that 17 of 25 learned familiar target passes and all seven OOD passes contradicted observed points. The reference fit all cubic observations but missed all their targets. Preserve target, observation and process checks as distinct fields; a Judge cannot remove this information limit by expressing confidence. Sequential decision learning remains a separately specified research question, not a synonym for confidence fitting.

The first Phase 2A.1 attempt was invalidated because trainer metadata contained holdout coefficient vectors. It was stopped before holdout scoring, retained, corrected and fully rerun without changing the protocol. Maintain the manifest allowlist and tests that prohibit training access to holdout/audit files. Eight familiar-case worked-model successes and two OOD successes still had incorrect process traces; final target agreement does not prove reasoning correctness.

In parallel with model planning, prepare the VIT hardware audit for an authorized lab operator. No lab access, IP inventory or measured throughput is available in this workspace. The current local multi-process result cannot establish 70–84-node feasibility. Use the template and protocol; keep missing fields null until measured.

## Findings to preserve

The first-stage experiment has mixed metrics; adding every stage did not dominate all simpler alternatives. The Judge is useful in its simple familiar family but overconfident under cubic shift. Later learned Researchers generate valid JSON while retaining poor numerical success. The staged curriculum did not improve its registered main metric, and earlier auxiliary formats deteriorated after task replacement. These are research signals, not results to suppress. Exact values, distinctions and paths are in the phase reports.

Phase 2B improved familiar proper scores with richer observed features, but one seed failed shifted utility despite zero incorrect forecasts at probability >=.95. Calibration-set temperature scaling did not consistently improve holdout scores. The study now includes cubic training worlds; quartic and mixture shifts were tested together. Do not describe its cubic cases as unseen or its offline policy utility as measured Controller savings. The failed development test and the failed evidence gate are both retained and distinguished.

Phase 2B.1 ran real tool dispatches with shared accounting. It preserved the three Phase 2B rich/log/temperature checkpoints; the reproduction requires their exact local files and sidecars. Its new family hides the difference from all four observed points by construction, so unconditional acquisition cannot pass the balanced stress utility condition at positive acquisition cost. Keep that mathematical limit explicit; do not present it as an unexpected empirical discovery. All 1,248 episodes and the development assertion failure were retained.

## Reproducibility and change control

Record configuration, source commit and dirty status, source/data/tokenizer/checkpoint hashes, seed, environment/hardware, stage parameters, all metrics and failures. Keep the existing public evaluation version unchanged; add reviewed versions for new tasks. Large datasets, weights and run directories remain local unless an explicit artifact publication plan is approved; portable result summaries and test logs are versioned in Git.

The owner requested regular descriptive commits and pushes to `main` at meaningful boundaries in [ashvyagni/AIM](https://github.com/ashvyagni/AIM). Preserve history and unrelated work; do not force-push. A completed phase report must state what actually works, tests run, benchmark scope, failed experiments, unresolved decisions, next experiment and exact changed files.

Major architectural alternatives remain experiments until evidence and owner review justify a change. Routine fixes and reversible implementation work may proceed within the approved direction. Do not claim a proprietary RLCD reproduction, a never-before-done invention, a VIT benchmark, or scientific capability without the corresponding evidence.

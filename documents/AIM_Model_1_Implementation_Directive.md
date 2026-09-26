# AIM Model-1 — implementation directive and handoff

**For the next implementation agent or ML engineer.** Updated 2026-09-26. Work in the existing `AIM/` workspace; the canonical code directory is `model-1/`. Continue the approved architecture and current implementation. This directive supplies a concrete starting point, not a request to redesign AIM.

**Latest state:** Phase 2C is implemented and build-validated. Read its [report](../model-1/reports/phase-2c-implementation.md) and [symbolic specification](../model-1/docs/SYMBOLIC.md). All 140 tests passed; the checker passed 18 cases and 48 real episodes were audited. The formula reference verified all eight supported expansions; the learned Researcher variants verified none. Keep learned models opt-in. The owner prioritizes the build phase: next implement corpus intake, resumable streaming batches and tokenizer comparison interfaces using small local allocations.

**Completed run:** `model-1/runs/20260926T181327-phase-2c-build-5e038999` completed in 63.918 seconds and exported to `model-1/reports/phase-2c-evidence/`. It contains separate SFT, preference, RLVR and symbolic Judge checkpoints, full tests and six evaluation variants. The RLVR+Judge variant failed generation before a Judge decision; successful learned-Judge integration was exercised with the formula proposer. Do not restart this run as unfinished work.

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
- Versioned state/claims/evidence, deterministic action budgets, bounded subprocess tools, exact numerical/provenance verifiers and append-only event replay.
- Regression tests, public evaluation fixtures, complete reproduction command, local CPU and Gloo/DDP benchmark infrastructure.

The 90,624-parameter initial decoder, 228,096-parameter structured Researcher and 225/289-parameter feature Judges are micro-scale mechanism checks. They are not final model/Judge size choices and do not replace the 100M–300M proxy → ~1B systems → conditional 7B+ roadmap. Do not represent the current system as a general research AI.

## First actions

```sh
cd model-1
.venv/bin/python -m unittest discover -s tests -v
```

Use the documented environment setup if the local `.venv` is missing. Read the latest report and existing bundle's `tests.log`, `results.json`, case records and export manifest before launching experiments. A missing torch environment is not a successful neural test result. The documented reproduction command can repeat a study if needed; do not rerun a completed study merely to begin the next phase.

Continue the [implementation plan](../model-1/docs/IMPLEMENTATION_PLAN.md) with the corpus/pretraining infrastructure at the start of Phase 3. Build license/provenance intake manifests, resumable streaming next-token batches, tokenizer comparison interfaces and small local continuation checks. Keep actual 100M–300M proxy training gated on data readiness and physical hardware evidence. No pretrained base imports or arbitrary scale selection.

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

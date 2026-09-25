# AIM Model-1 — implementation directive and handoff

**For the next implementation agent or ML engineer.** Updated 2026-09-25. Work in the existing `AIM/` workspace; the canonical code directory is `model-1/`. Continue the approved architecture and current implementation. This directive supplies a concrete starting point, not a request to redesign AIM.

**Latest state:** Phase 2A.2 is implemented, evaluated and audited. Read its [report](../model-1/reports/phase-2a2-implementation.md), [reproduction guide](../model-1/docs/ARITHMETIC_CURRICULUM.md) and [observation-scope note](../model-1/docs/OBSERVATION_SCOPE.md). All 92 preflight tests passed and 1,664 Controller cases ran. Worked/curriculum mean verified success was 6.77%/6.25%; both missed the capability gate and curriculum failed its advantage gate. Keep the deterministic default. Next implement separate Judge calibration and decision experiments under shift.

**Completed run:** `model-1/runs/20260925T073010-curriculum-reproduction-60d13ca4` has status COMPLETED and a final summary. Its selected checkpoints, 1,664 cases and observation audit were checked before export to `model-1/reports/phase-2a2-evidence/`. Do not restart it as unfinished work. All runs/checkpoints remain local; portable evidence is versioned in Git. The earlier [Phase 2A.1 report](../model-1/reports/phase-2a1-implementation.md) and [exact continuation audit](../model-1/reports/research-resume-audit.md) retain their historical results.

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
- Versioned state/claims/evidence, deterministic action budgets, bounded subprocess tools, exact numerical/provenance verifiers and append-only event replay.
- Regression tests, public evaluation fixtures, complete reproduction command, local CPU and Gloo/DDP benchmark infrastructure.

The 90,624-parameter initial decoder, 228,096-parameter structured Researcher and 225-parameter Judge are micro-scale mechanism checks. They are not final model/Judge size choices and do not replace the 100M–300M proxy → ~1B systems → conditional 7B+ roadmap. Do not represent the current system as a general research AI.

## First actions

```sh
cd model-1
.venv/bin/python -m unittest discover -s tests -v
```

Use the documented environment setup if the local `.venv` is missing. Read the latest report and existing bundle's `tests.log`, `results.json`, case records and export manifest before launching experiments. A missing torch environment is not a successful neural test result. The documented reproduction command can repeat a study if needed; do not rerun a completed study merely to begin the next phase.

Continue with [Phase 2B](../model-1/docs/IMPLEMENTATION_PLAN.md): register a separate Judge study before generating its holdout or tuning decisions. Define the forecast event, admissible pre-measurement features, proper-score and constant baselines, abstention/measurement utilities, calibration split, family shift and acceptance rules. Keep Researcher checkpoints and task contracts fixed. Reject features containing future measurements, hidden coefficients, outcome labels or post-verification state. All three Phase 2A world sets are exposed; use fresh partitions. Keep the deterministic Researcher as an explicit reference, not a hidden fallback.

Phase 2A.2 found that 17 of 25 learned familiar target passes and all seven OOD passes contradicted observed points. The reference fit all cubic observations but missed all their targets. Preserve target, observation and process checks as distinct fields; a Judge cannot remove this information limit by expressing confidence. Sequential decision learning remains a separately specified research question, not a synonym for confidence fitting.

The first Phase 2A.1 attempt was invalidated because trainer metadata contained holdout coefficient vectors. It was stopped before holdout scoring, retained, corrected and fully rerun without changing the protocol. Maintain the manifest allowlist and tests that prohibit training access to holdout/audit files. Eight familiar-case worked-model successes and two OOD successes still had incorrect process traces; final target agreement does not prove reasoning correctness.

In parallel with model planning, prepare the VIT hardware audit for an authorized lab operator. No lab access, IP inventory or measured throughput is available in this workspace. The current local multi-process result cannot establish 70–84-node feasibility. Use the template and protocol; keep missing fields null until measured.

## Findings to preserve

The first-stage experiment has mixed metrics; adding every stage did not dominate all simpler alternatives. The Judge is useful in its simple familiar family but overconfident under cubic shift. Later learned Researchers generate valid JSON while retaining poor numerical success. The staged curriculum did not improve its registered main metric, and earlier auxiliary formats deteriorated after task replacement. These are research signals, not results to suppress. Exact values, distinctions and paths are in the phase reports.

## Reproducibility and change control

Record configuration, source commit and dirty status, source/data/tokenizer/checkpoint hashes, seed, environment/hardware, stage parameters, all metrics and failures. Keep the existing public evaluation version unchanged; add reviewed versions for new tasks. Large datasets, weights and run directories remain local unless an explicit artifact publication plan is approved; portable result summaries and test logs are versioned in Git.

The owner requested regular descriptive commits and pushes to `main` at meaningful boundaries in [ashvyagni/AIM](https://github.com/ashvyagni/AIM). Preserve history and unrelated work; do not force-push. A completed phase report must state what actually works, tests run, benchmark scope, failed experiments, unresolved decisions, next experiment and exact changed files.

Major architectural alternatives remain experiments until evidence and owner review justify a change. Routine fixes and reversible implementation work may proceed within the approved direction. Do not claim a proprietary RLCD reproduction, a never-before-done invention, a VIT benchmark, or scientific capability without the corresponding evidence.

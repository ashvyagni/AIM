# AIM Model-1 — implementation directive and handoff

**For the next implementation agent or ML engineer.** Updated 2026-09-25. Work in the existing `AIM/` workspace; the canonical code directory is `model-1/`. Continue the approved architecture and current implementation. This directive supplies a concrete starting point, not a request to redesign AIM.

**Latest state:** Phase 2A.1 is implemented and evaluated. Read its [report](../model-1/reports/phase-2a1-implementation.md), [reproduction guide](../model-1/docs/FINITE_DIFFERENCE_COMPARISON.md) and [exact continuation audit](../model-1/reports/research-resume-audit.md). All 77 tests passed and 1,664 Controller cases ran. Worked supervision improved mean verified success from 5.21% to 12.50%, but one seed regressed and both arms missed the capability gate. Keep the deterministic default. The next proposed model experiment is a preregistered arithmetic curriculum, not a larger model or joint reward.

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
- A strict neural Researcher adapter; current arithmetic-trained weights fail hypothesis generation and that result is recorded.
- Checkpoint-versioned plain/worked hypothesis formats, fresh world partitions, frozen multi-seed comparisons, independent process diagnostics and tested Researcher continuation.
- Versioned state/claims/evidence, deterministic action budgets, bounded subprocess tools, exact numerical/provenance verifiers and append-only event replay.
- Regression tests, public evaluation fixtures, complete reproduction command, local CPU and Gloo/DDP benchmark infrastructure.

The 90,624-parameter initial decoder, 228,096-parameter structured Researcher and 225-parameter Judge are micro-scale mechanism checks. They are not final model/Judge size choices and do not replace the 100M–300M proxy → ~1B systems → conditional 7B+ roadmap. Do not represent the current system as a general research AI.

## First actions

```sh
cd model-1
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m aim.reproduce
```

Use the documented environment setup if the local `.venv` is missing. Read the new bundle's `tests.log`, `index.json`, metrics, manifest and failures. It uses no pretrained assets or online service. A missing torch environment is not a successful neural test result.

Phase 2A.1 has completed the numerical/worked-process comparison and continuation work. Continue with the proposed [Phase 2A.2](../model-1/docs/IMPLEMENTATION_PLAN.md): register an arithmetic-curriculum ablation, compare training mastery with generalization, and use fresh coefficient worlds excluding both previous experiments. Both historical tests are now exposed. A separately versioned observation-consistency verifier is also recommended. Keep the deterministic Researcher as an explicit reference, not a hidden fallback.

The first Phase 2A.1 attempt was invalidated because trainer metadata contained holdout coefficient vectors. It was stopped before holdout scoring, retained, corrected and fully rerun without changing the protocol. Maintain the manifest allowlist and tests that prohibit training access to holdout/audit files. Eight familiar-case worked-model successes and two OOD successes still had incorrect process traces; final target agreement does not prove reasoning correctness.

In parallel with model planning, prepare the VIT hardware audit for an authorized lab operator. No lab access, IP inventory or measured throughput is available in this workspace. The current local multi-process result cannot establish 70–84-node feasibility. Use the template and protocol; keep missing fields null until measured.

## Findings to preserve

The first-stage experiment has mixed metrics; adding every stage did not dominate all simpler alternatives. The Judge is useful in its simple familiar family but overconfident under cubic shift. The neural Researcher diagnostic fails structured output. These are research signals, not results to suppress. Exact values and paths are in the implementation report.

## Reproducibility and change control

Record configuration, source commit and dirty status, source/data/tokenizer/checkpoint hashes, seed, environment/hardware, stage parameters, all metrics and failures. Keep the existing public evaluation version unchanged; add reviewed versions for new tasks. Large datasets, weights and run directories remain local unless an explicit artifact publication plan is approved; portable result summaries and test logs are versioned in Git.

The owner requested regular descriptive commits and pushes to `main` at meaningful boundaries in [ashvyagni/AIM](https://github.com/ashvyagni/AIM). Preserve history and unrelated work; do not force-push. A completed phase report must state what actually works, tests run, benchmark scope, failed experiments, unresolved decisions, next experiment and exact changed files.

Major architectural alternatives remain experiments until evidence and owner review justify a change. Routine fixes and reversible implementation work may proceed within the approved direction. Do not claim a proprietary RLCD reproduction, a never-before-done invention, a VIT benchmark, or scientific capability without the corresponding evidence.

# Phase 3D — Tokenizer-compatible post-training build

Date: 2026-09-27. **Status: implementation written; execution and formal audit deferred by owner request.**

## Completion assessment

The AIM codebase is not complete as a general research model or production training platform. Previous phases established working miniature research loops, separate training/Judge objectives, exact scoped verifiers, provenance, local distributed pretraining and corpus/operator preparation. Learned research capability, representative data, physical VIT scaling, large training, long-context retrieval, broader verified tools and production operations remain unfinished.

This phase implements the next bounded model-stack upgrade. It does not replace the approved architecture or claim that all remaining work can be settled by adding code.

## Built

- An opt-in language checkpoint schema binding full tokenizer specifications, response encoding, model/task/output contracts and lineage.
- Byte/BPE prompt-response batching with explicit boundary handling, response masks, context rejection and per-example encoding records.
- Train/validation-only inputs for arithmetic, symbolic, structured numerical SFT and explicitly attributed external text/preferences.
- Separate tokenizer-aware SFT, DPO and finite-candidate RLVR stages with frozen references, periodic checkpoints and strict resume identities.
- A BPE-aware pretraining initialization path; no implicit embedding/vocabulary remapping.
- Versioned runtime loading, structured numerical/symbolic adapter contracts and strict greedy generation records.
- Checkpoint descriptions, non-executing stage plans, four CLI command groups and four bounded example configs.
- Candidate unit/integration tests and a deferred byte/BPE training/continuation reproduction runner.

Read the [complete implementation specification and commands](../docs/TOKENIZED_POSTTRAINING.md).

## What actually works / evidence limits

**New runtime behavior has not been executed or validated in this phase.** No tests, training experiments, generation runs, performance benchmarks or formal audit were run, following the owner's request to reserve those for the next phase. Test files and the audit runner are prepared, not passed evidence. The previous 192-test result belongs to Phase 3C and must not be attributed to this source revision.

The code is opt-in through `language-train` and the new checkpoint kind. Existing legacy objective implementations and default Controller/Judge policies are retained. Shared entry points gained conditional dispatch for the new kind, so the next audit must include legacy regressions too. No capability or default-model promotion is justified yet.

## Decisions and remaining risks

Prompt and response are encoded separately to prevent BPE merges across the loss boundary. Exact tokenizer specification, rather than vocabulary size, identifies compatibility. Initialization resets optimizer/reference; resume restores state and requires unchanged objective/data/encoding/configuration except a higher step target. All learned weights retain AIM's own random-init lineage.

SFT, preference, RLVR and Judge calibration remain separate. Procedural preference labels are not human RLHF. RLVR remains a bounded candidate environment, not general reasoning RL. No combined reward, PPO, learned language Judge, model-size promotion, MoE or hardware strategy change was introduced.

Pending risks include encoding/masking bugs, checkpoint validation gaps, task-transition mistakes, reference/RNG continuation errors, malformed generation handling, legacy adapter regressions and incomplete external-data boundary enforcement. These must be evaluated by executing and reviewing the candidate tests, not by assuming correctness from the design.

## Next audit sequence

1. Read the new specification and inspect the changed modules against task/tokenizer/checkpoint invariants.
2. Run `test_language_pipeline.py`; fix failures while retaining logs and failed run directories.
3. Run the complete `aim.language_reproduce` protocol into a new result directory. Require no skips, exact continuation and matching frozen references across the byte/BPE stages.
4. Exercise structured and symbolic Researcher adapters in actual Controller episodes, plus the distributed initialization bridge and periodic-checkpoint interruption recovery.
5. Re-run the legacy regression suite and compare unchanged canonical evaluations.
6. Publish actual results, failures, exact source identity and a reviewed readiness decision. Keep larger scale and useful research capability separate from infrastructure correctness.

Continue the audit when the owner resumes it; the prepared runner does nothing merely by existing.

## Files and repository delivery

The [changed-file inventory](phase-3d-files.json) lists this build against baseline `7e9589a`, excluding the inventory itself. Code/contracts, trainer/data modules, runtime/configs and pending tests/docs are committed in focused groups to the owner's `main` branch. Earlier result artifacts remain historical evidence.

The [machine-readable status](phase-3d-status.json) marks this build BUILT_UNVALIDATED and separates the previous test count from pending work.

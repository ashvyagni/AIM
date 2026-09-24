# Model-1 decision record

Recorded 2026-09-24. This document reconciles the existing research baseline with later explicit owner instructions. It does not retrospectively claim that prototype choices were scientifically settled in the handbook.

## D001 — Preserve the modular architecture

**Status: owner-approved direction, implemented as interfaces.** Researcher, separate Judge, deterministic Controller, independent Verifiers and provenance Memory remain distinct. Alternatives such as shared heads, MoE or an extra learned critic require controlled experiments and owner review. The miniature implementation does not add a mandatory third learned model.

Rationale: separate ownership allows direct tests of confidence versus evidence, independently trained objectives and replaceable backends. Cost: more contracts, orchestration and eventually component latency. The current evidence establishes the interfaces' operation, not superiority over every alternative.

## D002 — All learned weights start from scratch

**Status: explicit later owner requirement, implemented.** Earlier documents recommended starting from an appropriately licensed pretrained checkpoint. The owner later rejected that direction. Model-1 initializes its own transformer and Judge. Using standard numerical/training libraries remains allowed. No vendor/model novelty claim follows from random initialization alone.

Trade-off: ownership and auditable initialization versus substantially more data, compute and training research needed to achieve useful language capability. The present micro-model cannot perform general research. A future pretrained comparison would require explicit approval and must not replace the from-scratch project silently.

## D003 — Follow the latest stage order

**Status: explicit latest mission, implemented as separable stages.** SFT → preference optimization → verifier-grounded RLVR → separate Judge calibration → future integrated loop training. Older research tables with a different order remain historical; they do not override the latest request.

SFT→RLVR is a comparison branch, not an undocumented change to the primary order. No weighted RLHF+RLVR+RLCD scalar is defined. A combined method remains a hypothesis requiring mathematical definition and controlled evidence.

## D004 — A micro-model precedes 100M–300M proxies

**Status: reversible engineering choice.** The 90,624-parameter transformer and 225-parameter feature Judge minimize test costs while establishing actual weight updates and interfaces. The 2M guard prevents accidental large allocation in this runner. These numbers do not replace the owner's 100M–300M proxy / ~1B systems / 7B+ conditional roadmap.

Judge scale is unresolved for language research tasks. This phase supplies no evidence for a fixed 200M/1B/etc. Judge. Test feature classifiers, small independent language Judges and other approved candidates on calibrated decisions per unit compute before deciding.

## D005 — Deterministic Researcher is an explicit reference backend

**Status: engineering baseline, not final product.** Polynomial fitting supplies a predictable positive and negative control for the system loop. The native transformer adapter exists, but the current arithmetic SFT checkpoint fails to emit valid hypothesis JSON. No fallback silently substitutes the reference and labels the result neural.

Next experiment: train structured hypothesis generation using project-generated, verifier-labeled traces and evaluate held-out worlds and families. Keep the reference backend for regression and comparative evaluation even after a learned backend works.

## D006 — DPO is the first preference-learning implementation

**Status: prototype recommendation.** DPO offers an executable pairwise objective with a fixed reference and direct gradient tests. Actual human-feedback collection, explicit reward-model fitting and PPO are deferred. Included labels are procedural. See [training specification](TRAINING.md) for the primary source and exact objective.

Evidence is limited to mechanism correctness and tiny observed metrics; the phase does not establish DPO as the best large-model alignment method.

## D007 — RLVR begins with an exact bounded environment

**Status: prototype scope choice.** Three-candidate arithmetic REINFORCE exercises sampled rewards, a detached baseline, a fixed reference and KL logging. Candidates include the exact correct answer by construction. This cannot substantiate a reasoning capability claim. Open-ended generation, process rewards and broader verifier ensembles need new experiments and reward-hacking tests.

## D008 — RLCD-style calibration is a defined local method

**Status: operational definition; proprietary equivalence unresolved.** The implemented Judge learns a pre-measurement event with proper scoring, followed by validation-only temperature selection. It is supervised calibration, not a claim to reproduce TypeSafe's proprietary RLCD or Jev training.

The existing literature matrix's `jevtypesafeai.com` entry must not be treated as an authenticated official technical report. Primary company material should be attributed to [TypeSafe's own domain](https://typesafe.ai/blog/introducing-system-one-models-and-jev), and product descriptions do not establish the complete training algorithm. Historical research files are preserved; this source qualification supersedes any “official” label applied to an unverified third-party domain.

## D009 — Verification gates truth status

**Status: implemented invariant.** Only independent scoped checks establish VERIFIED. A trained Judge's confidence does not. Unknown verification outcomes are not false labels. The recorded out-of-family calibration failure reinforces the value of keeping the gate, but does not prove a universal calibration solution.

## D010 — Dense decoder and byte tokenizer are reference implementations

**Status: provisional engineering choice.** RMSNorm/RoPE/SwiGLU/GQA/tied embeddings use ordinary PyTorch operations. Manual KV repetition enables the CPU reference without relying on native GQA backend availability. Byte tokenization avoids borrowed tokenizer artifacts. Optimized attention, cache, tokenizer training, context extension, MoE and sharding remain experiments.

The architectural mechanisms are not claimed as AIM inventions. The potential research contribution lies in a measured improvement from the specified system/training design, if future experiments establish one.

## D011 — Preserve evidence and failed runs

**Status: implemented.** Unique run directories, source snapshots, configuration/data/checkpoint hashes, event history and failure records are written locally. Canonical evaluation versions are immutable by process. Runs are excluded from Git by default, retained on disk, and summarized in portable reports. The repository was initially absent; early records correctly have null Git identity. Later commits do not retroactively rewrite those manifests.

On the owner's explicit request, code and documentation are committed and pushed to `main` at meaningful phase boundaries. No force push or rewriting of failed evidence is authorized by that request.

## D012 — VIT scale remains hardware-dependent

**Status: unresolved until measured.** The owner reports 70–84 i9/32GB/UHD770 desktops. Access, network, sustained throughput and aggregate usable budget are not established. Local Mac and loopback results are useful implementation checks only. No final parameter count, training duration or cluster speedup is selected from them.

## How to propose a change

Create a new numbered record with: affected invariant; proposed alternative; primary evidence; competing choices; expected benefit/cost; predeclared experiment; metrics and stopping rules; actual results including failures; and owner review status. Keep current architecture until sufficient evidence and appropriate approval support a change. Hypotheses and unresolved questions must retain those labels.

## D013 — Phase 2A stays experimental after failing its gate

**Status: evidence-backed engineering decision, 2026-09-24.** A checkpoint-versioned structured contract, programmatic world-disjoint SFT and three random seeds were implemented without changing the dense architecture or component responsibilities. The new contract binds model-generated evidence aliases to real spans and rejects malformed outputs. The default reference backend and phase-1 canonical suite remain unchanged.

The [registered experiment](experiments/phase-2a-protocol.md) required at least 90% valid output, 25% independently verified success and a 10-percentage-point gain over initialization for each seed. Held-out success was 7/64, 2/64 and 2/64, despite 95–100% contract validity. None reached the success threshold. Zero unbacked VERIFIED claims were observed in 896 loop executions. See the [complete result](../reports/phase-2a-implementation.md).

Decision: retain the deterministic default and keep learned checkpoints opt-in. Do not promote low token loss or valid syntax to a reasoning-success claim. Next compare explicit numerical/process supervision under a new predeclared protocol with fresh world holdouts. The existing separate Judge and training-stage thesis are unchanged; no larger scale or combined reward is approved by this result.

# AIM Model-1 — Phase 1 implementation report

**Date:** 2026-09-24. **Scope:** foundation and first end-to-end miniature research-loop prototype.

**Tested implementation commit:** `a1987adc13a762efa533d9e00eee5122e84a9e74`. Documentation and portable evidence were added after that code commit. The full run manifest also records the untracked documentation present at execution; this is not represented as a clean-tree run.

## Result

A runnable modular foundation is implemented in `model-1/`. The reference Researcher, separate Judge, deterministic Controller, numerical/provenance Verifiers and local evidence Memory complete an investigation from question to scoped, traceable final claims. A native transformer and independent Judge are actually trained from random initialization with separate objectives.

**43 tests passed, 0 skipped.** Both rule-Judge and trained-Judge evaluations passed **8/8** public engineering cases. A two-process local DDP run verified collective results and identical replica parameters. These results establish the miniature implementation and interfaces, not general research intelligence or VIT cluster feasibility.

The learned Researcher is not yet useful: the current arithmetic SFT checkpoint emits invalid hypothesis JSON. The working research loop uses the explicitly named deterministic reference Researcher. The trained Judge has a serious out-of-family calibration failure, retained below.

## Evidence artifacts

- [Portable machine-readable results](evidence/phase-1-results.json): exact metrics, source run identifiers, tested commit, environment, manifest/checkpoint hashes and distributed comparison.
- [Complete final test log](evidence/tests.log): all 43 cases and actual execution result.
- [Integrated example state](evidence/example-state.json): hypotheses, claims, pre-measurement forecasts, scoped checks, evidence spans and final response text. Relative object links inside this copied JSON refer to the original run's memory directory.
- [Initial provenance failure traceback](evidence/initial-provenance-failure.log) and [initial distributed launch failure](initial-ddp-launch-failure.md).
- [Exact changed-file inventory](phase-1-files.json).
- Full local reproduction bundle: `model-1/runs/20260924T130658-reproduction-6e3849a2/`; its `index.json` identifies every experiment. Source snapshots, datasets, checkpoints, SQLite memory and event history remain there and are excluded from Git.

Portable summaries and logs are committed. A fresh clone can reproduce new run bundles using the commands below; it does not contain the original binary checkpoints. Earlier runs remain intact, including ones created before Git initialization with a truthful null commit field.

## What was built

| Area | Delivered mechanism |
|---|---|
| Component contracts | Researcher, Judge and Verifier protocols; bounded actions/results; explicit ResearchState |
| Researcher | Numerical reference backend and own-checkpoint neural adapter with strict JSON contract |
| Judge | Rule baseline; separate 225-parameter feature MLP with proper-score training and temperature fitting |
| Controller | Deterministic phases, evidence/action identity validation, deadlines, action budget, abstention and final verifier gates |
| Memory/provenance | Versioned source objects, exact spans and hashes, SQLite graph edges, append-only event records and replay |
| Verification | Exact numeric agreement and source-span integrity; explicit PASS/FAIL/UNKNOWN/ERROR/TIMEOUT |
| Native model | 90,624-parameter dense causal decoder, byte tokenizer, RMSNorm/RoPE/SwiGLU/GQA/tied embeddings |
| Training | Response SFT, DPO, bounded sampled RLVR, separate Judge learning; native checkpoints and exact CPU resume |
| Data pipeline | Procedural fixtures and provenance-aware external SFT/preference intake; no real human labels collected |
| Evaluation | Unit/integration/boundary/gradient/resume tests; versioned eight-case engineering suite; proper-score metrics |
| Experiment records | Unique retained runs, config/input/source hashes, source archives, environment, parent checkpoints, metric events and failure tracebacks |
| Hardware | Local CPU benchmark, bounded loopback launcher, Gloo/DDP rank benchmark and unfilled physical lab audit template |
| Handoff | Setup, architecture, training/evaluation specs, decisions, hardware protocol and phased next-agent directive |

See the [sixteen-area coverage matrix](../docs/IMPLEMENTATION_PLAN.md) for explicit production and research work still pending.

## End-to-end investigation actually observed

For the fixture with observed points (0,1), (1,4), (2,9), the reference Researcher predicts at x=4:

- Linear candidate: 13; independent measurement: 25 → **CONTRADICTED**.
- Quadratic candidate: 25; independent measurement: 25 → **VERIFIED at that x and tolerance**.

Forecasts are written before measurement. Confidence does not establish either status. The memory preserves the rejected candidate, check records and exact source spans. The final response explicitly states that matching one measurement does not prove the global law.

## Tests run

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m aim.reproduce
```

The final reproduction reran all 43 tests with the pinned torch environment and no skips. Coverage includes changed sources, forged citations, event tampering, state-copy boundaries, wrong-claim Judge output, confidence failing to override a verifier, unavailable/conflicting evidence, exhausted budgets, actual worker termination on a tiny deadline, zero-input polynomial calculation, nonfinite inputs, unresolved labels, canonical-suite integrity, failed-run retention including missing input, Unicode round trip, causal attention, masking/padding, analytical parameter counts, DPO/RLVR gradient directions, frozen references, split independence, checkpoint corruption and exact CPU continuation.

The suite is an engineering regression suite, not a concealed scientific test set. Intentional failure tests use isolated temporary fixtures; real experiment directories and their actual failures are retained.

## Training observations

All rows below use seed 17, the same 90,624-parameter transformer, a tiny procedural training set and a 32-example validation subset. Preferences are programmatic. The candidate reward is restricted to three environment-supplied answers, one of which is correct by construction.

| Checkpoint | Response-token NLL ↓ | Preference ordering accuracy ↑ | Bounded expected reward ↑ |
|---|---:|---:|---:|
| Random initialization | 5.4980 | 0.5625 | 0.3677 |
| SFT, 60 updates | 1.5826 | 0.4063 | 0.2622 |
| SFT → DPO, 30 more updates | 1.9085 | 0.4688 | 0.2431 |
| SFT → DPO → RLVR, 30 more updates | 1.8991 | 0.4375 | 0.2595 |
| SFT → RLVR, 30 more updates | 1.7294 | 0.4063 | 0.2749 |

**Interpretation:** updates and separate objectives operate, but adding stages does not monotonically improve these metrics. SFT lowers token NLL while reducing the bounded candidate reward relative to random initialization; the latter starts with a small action set and is not a meaningful measure of language competence. DPO improves one ordering metric over SFT but worsens others. The integrated sequence is not established as better than SFT→RLVR. Different total training budgets, one seed, tiny samples and inspected procedural data prevent a scientific superiority conclusion.

The neural Researcher diagnostic returned zero claims with `Researcher emitted invalid hypothesis JSON: ''`. The run completes with an unresolved result, and does not substitute the reference Researcher. Structured research-trace training is the next concrete model task.

## Judge calibration observations

The 225-parameter Judge was independently initialized and trained on 256 synthetic forecast records. Temperature T=0.5 was selected using only the 96-record validation split. Test and OOD splits each contain 96 candidate records, grouped by distinct underlying worlds.

| Split | Initial Brier ↓ | Final unscaled Brier ↓ | Temperature-scaled Brier ↓ | Scaled log loss ↓ |
|---|---:|---:|---:|---:|
| Familiar linear/quadratic test | 0.25365 | 0.01619 | 0.01216 | 0.03982 |
| Unseen cubic family | 0.24111 | 0.51083 | **0.52277** | **5.29611** |

On the OOD records, threshold 0.95 retains about 51.04% of predictions and every retained prediction fails the measurement check in this diagnostic. Calibration fitted to the familiar family worsens shifted-family overconfidence. This failure is retained, not treated as a deployment-ready Judge. No test/OOD performance was used to select temperature.

The runtime learned-Judge suite still passes 8/8 because the default policy verifies every candidate and the Controller obeys independent check outcomes. Its eight resolved forecasts have Brier 0.14434 and log loss 1.52987; passing the loop regression does not mean the Judge's forecasts are well calibrated. Missing measurements remain unscored.

## Benchmark results actually measured

**Host:** local macOS arm64, 8 reported logical CPUs, FP32 CPU, one torch thread per process. CPU marketing-name lookup unavailable; recorded null. **No VIT machine was accessed.**

| Measurement | Observed result | Scope |
|---|---:|---|
| Tiny complete training update | Median 3.296 ms; 38,229.6 processed tokens/s | 90,624 parameters; batch 2; 63 prediction tokens per example; 2 warmups + 5 samples |
| One-rank DDP | 23,249.6 tokens/s | Batch 2, sequence 32, one local process |
| Two-rank DDP | 36,714.9 aggregate tokens/s | Same per-rank work; two processes on the same physical Mac |
| Local process comparison | 1.579× throughput; 78.96% of ideal two-process throughput | Weak-scaling diagnostic; shared host, five samples |
| Two-rank all-reduce | Median 0.354 ms for 262,144-byte payload | Loopback, not lab Ethernet |
| Replica consistency | Exact parameter hashes match | After actual DDP optimizer updates |
| Checkpoint round trip | Equal reopened model tensors | Local file write/read; no durable-fsync claim |

The single-process CPU microbenchmark and DDP workloads use different context lengths and should not be compared as a speedup pair. Repeated microbenchmark timings varied across runs; all samples remain stored. No sustained thermal behavior, accelerator speed, physical-network bandwidth, multi-host efficiency, large-model memory or training completion estimate was measured.

## Failures and limitations retained

1. **Initial loop failure:** nested dataclass serialization failed while constructing verifier provenance. Retained in `runs/20260924T072249-loop-9f9a1417/`. Recursive canonical serialization fixes it; regression tests cover it.
2. **Dependency setup interruption:** initial restricted-network downloads failed, then torch download was interrupted by connectivity. The later authorized pinned installation succeeded. No training result was attributed to the failed installation attempts; the interaction transcript retains the installer errors.
3. **Initial local distributed launch:** standalone rendezvous failed to resolve/connect to `localhost:0`, retried and exited before rank scripts began. The incident is recorded separately. A bounded IPv4 loopback launcher then succeeded; the failed attempt has no performance result.
4. **Missing-input record gap found during review:** run construction previously hashed absent inputs before entering failure recording. The fix records a null hash and allows the actual read to fail inside the retained run; its new regression passes.
5. **Neural Researcher capability failure:** malformed/empty structured output on the research fixture. Not corrected by pretending the deterministic baseline is learned.
6. **Judge distribution-shift failure:** severe OOD overconfidence, worsened by familiar-domain temperature scaling.
7. **Staged training does not dominate:** mixed validation diagnostics do not justify a breakthrough claim or a final objective design.

## Decisions and recommended next experiment

Maintain the approved component separation and verifier gate. Keep the learned Judge as an experimental forecaster, with all predictions checked. Do not authorize a 1B/7B training commitment from these results.

Next implement a structured Researcher dataset and training experiment with whole-world/family holdouts, then compare reference, random and trained native Researcher outputs through the same loop. Independently collect an authorized physical VIT inventory and 1/2/4/8-host measurements. Judge shift calibration and matched-budget multi-seed objective ablations follow the predeclared plan.

Unresolved: final Researcher/Judge scale, tokenizer and large-corpus mixture, robust OOD calibration, open-ended RLVR, process supervision, long-horizon RLCD formulation, multi-host sharding, long-context retrieval, scientific evaluation and any defensible novelty claim. These remain explicit in the [implementation plan](../docs/IMPLEMENTATION_PLAN.md).

## Files and Git history

[phase-1-files.json](phase-1-files.json) lists every newly created implementation/documentation/support file and its hash, with its own entry excluded from recursive hashing. Existing research files and PDF were preserved unchanged and imported as the baseline. The old empty `AIM Model 1/` folder was not removed.

Pushed phase commits:

- `69f4f39` — `docs: preserve AIM research architecture baseline`
- `cf66071` — `feat: implement modular Model-1 research and training foundation`
- `a1987ad` — `fix: retain experiment failures for missing input files`

The final documentation/evidence commit follows these. Runs and `.venv` are intentionally untracked; no failed run was deleted. The next engineer should start with [the implementation directive](../../documents/AIM_Model_1_Implementation_Directive.md) and [Model-1 README](../README.md).

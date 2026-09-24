# Phase 2A.1 — finite-difference supervision comparison

Preregistered 2026-09-25 (local time), before generating or evaluating this experiment's holdout. Experiment version: `aim-finite-difference-comparison-v1`. Keep Phase 2A immutable. Tested resume is an engineering prerequisite; this experiment is a new random-initialization run, not further training selected on the inspected Phase 2A test.

## Question and evidence status

**Hypothesis:** explicitly supervising intermediate finite differences improves the small Researcher's numerical generalization. This is unproven at our scale. Phase 2A learned structure and intercept copying much better than slope/curvature. Keep the same 228,096-parameter dense decoder, byte tokenizer, separate rule Judge, deterministic Controller, provenance memory and two final verifiers.

Prior research distinguishes outcome and step-level supervision; Lightman et al. train a process reward model with human labels ([Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)). Wei et al. study intermediate-step prompting in large models ([Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)). Neither establishes that our tiny randomly initialized decoder benefits from programmatic step SFT. This experiment trains no reward model and makes no RLHF/RLVR/RLCD integration claim.

## Data before outcomes

Use the same coefficient universe as Phase 2A: c0,c1 integers -9..9, c2 -3..3; cubic OOD adds nonzero c3 in {-2,-1,1,2}. Exclude the actual complete coefficient vectors in **every** Phase 2A split, including train, validation, test and OOD. Compare coefficient vectors, not experiment-prefixed hashes. Record the exclusion list and its hash.

For each in-family stratum, sort remaining worlds by SHA-256 of the new experiment version and complete coefficient vector. Allocate consecutive nonoverlapping ranges: train 128 linear + 384 quadratic; validation 16 + 48; test 16 + 48. Use the first 64 remaining cubic worlds by that ranking for OOD. Rank allocation replaces old hash buckets because the remaining linear pool is smaller. Assert zero coefficient overlap within this experiment and with Phase 2A. Keep any cross-family observation ambiguity and report it.

Observe x=0,1,2, select target from 3,4,5 by world hash. Both arms have identical prompts, worlds and sampler sequence for each seed. Store real sources, exact spans, label origin, hidden truth and hashes. Trainer reads only the train/validation file. It receives no test/OOD labels.

## Arms and objective

Plain arm uses the existing exact output schema: `{"coefficients":[[c0,c1,c2]],"evidence":["E0"]}`.

Worked arm uses checkpoint contract `aim-finite-difference-output-v1`: `{"d1":d1,"d2":d2,"coefficients":[[c0,c1,c2]],"evidence":["E0"]}`. Teachers compute d1=y1-y0 and d2=y2-2*y1+y0, then c2=d2/2, c1=d1-c2 and c0=y0. The decoder generates all numbers. No arithmetic answer is injected into its context or substituted for its generation.

Both optimize ordinary response-token cross-entropy including EOS:

L(theta) = - sum_(i,t in response_i) log p_theta(z_it | prompt_i,z_i,<t) / sum_i |response_i|.

Intermediate tokens enter the worked arm's response objective. There are no auxiliary rewards, weighted reward mixture, constrained decoding, reference fallback or best-of-N candidates.

## Matched budget and known differences

Seeds 17,23,41, all weights independently random per seed; each pair starts with identical tensors. Model width96, layers2, heads4, KV2, FFN256, context256. Deterministic CPU FP32, one thread, AdamW LR0.001, weight decay0.01, clip1. Batch16, 1800 updates. Pad **both** arms' training inputs to exactly256 positions, including trailing masked padding. Thus each arm/seed processes 7,372,800 training input positions with identical dense forward/backward tensor shapes and identical sampled world sequence.

This is a matched dense-shape compute proxy and equal world-exposure budget, not measured equal FLOPs or equal wall time. Worked supervision intentionally contains more target tokens; report response-token counts, processed positions, elapsed time, inference output lengths and loop latency. Padding wastes compute but makes this bounded comparison easier to interpret. Validation generation cost may differ. No claim that information quantity is controlled.

Greedy generation cap128 tokens for both arms. Evaluate step0,300,600,1200,1800 on validation only. Select per arm/seed by prediction agreement, validity, lower response NLL, earliest step. NLL selects within an arm only; never compare NLL across different response schemas as a quality ranking. Deadline1200 seconds per seed/arm; retain emergency state/failures and mark any incomplete experiment.

## Frozen evaluation and diagnostics

Freeze hashes for all six selected models and six initial checkpoints before opening new holdout. Compare selected and initial checkpoints plus the single-candidate deterministic reference using actual Controller/tool/provenance/measurement runs, rule Judge and eight-action budget. Fixed test64 and OOD64 are shared across arms/seeds; do not pool them as independent tasks.

Report per-family valid output, final VERIFIED/CONTRADICTED/UNKNOWN counts, Wilson95 intervals, paired gained/lost worlds versus initial and versus the other arm, and seed dispersion. Every invalid output remains in the denominator. Log raw text and source-linked state, scope of checks, rejected citations and unbacked VERIFIED claims.

Independently diagnose d1,d2 and each coefficient. Wrong intermediate arithmetic remains a recorded process failure even if final prediction passes. Process diagnostics **do not** add a third gate to Controller verification in this experiment; final VERIFIED continues to mean only existing provenance and target-measurement checks. For cubic OOD, coefficient-vector recovery against hidden cubic coefficients is inapplicable to a degree-2 output; report observed-step consistency and final prediction outcomes separately.

## Predeclared decision rule

Each arm must independently meet the Phase 2A gate on **each** seed: >=90% contract-valid, >=25% final verified, >=10 percentage-point improvement over its own initial model, and zero unbacked VERIFIED claims across all runs. A worked-arm advantage additionally requires >=5 percentage points improvement in mean verified test rate over plain and no seed worse than plain. These are provisional engineering thresholds, not statistical significance or general reasoning claims. Report paired counts and uncertainty even when thresholds pass.

A gate miss is a result. Keep the deterministic default. Do not increase model size, extend training, tune decoding, change thresholds or reuse inspected holdout to rescue this experiment. Any successor needs a new protocol and data boundary. OOD has no success threshold. No conclusion about VIT training capacity follows from this local CPU run.

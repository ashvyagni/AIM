# Phase 2A protocol — structured native Researcher

Status: declared before training or evaluating the new holdout. Version `aim-structured-research-v1`, 2026-09-24. This is a bounded experiment within the approved component architecture.

## Question and hypothesis

Can the existing randomly initialized dense decoder learn to emit a supported, structured polynomial hypothesis that survives the unchanged Controller and independent measurement verifier? Hypothesis: supervised research traces improve valid output and verified prediction rates over identical untrained weights. This does not test general scientific discovery.

## Fixed data and interface

- In-family worlds: integer polynomials c0+c1*x+c2*x² with c0,c1 in [-9,9], c2 in [-3,3]. c2=0 is the linear/constant stratum; all other values form the quadratic stratum.
- Hash the canonical complete coefficient vector with this experiment's version prefix. Buckets 0–5 are train, 6–7 validation, 8–9 test. Rank deterministically within each stratum. Use 128 linear + 384 quadratic training worlds; 16 linear + 48 quadratic validation worlds; 16 linear + 48 quadratic test worlds. No coefficient world may cross a split.
- Observe x=0,1,2; target x is deterministically selected from 3,4,5. The target measurement and hidden coefficient vector are never in the Researcher prompt.
- OOD: 64 distinct cubic worlds, nonzero c3 in {-2,-1,1,2}, selected deterministically from a separate versioned pool. These labels are not used for training, checkpoint selection or tuning. Existing phase-1 cubic fixtures remain historical diagnostics, not this new holdout.
- Prompts contain structured observations under local evidence aliases (E0, etc.) and the target x. Aliases map to actual immutable source-span IDs. The model must emit exactly one candidate as `{"coefficients":[[c0,c1,c2]],"evidence":["E0"]}`. Invalid JSON, fabricated aliases or illegal coefficients fail closed; there is no reference-backend fallback or constrained decoding.
- Labels are generated from the known synthetic world and exact arithmetic. The original source, source hash, prompt, response, coefficient-world group, split, generator and label origin are retained. These are programmatic SFT traces, not human feedback or literature data.

## Models, budget and selection

Use the existing dense decoder and byte tokenizer, width 96, 2 layers, 4 query heads, 2 KV heads, FFN 256, context 256. Count parameters analytically and from actual tensors; stay below the existing 2M allocation guard. Train all weights from random initialization independently for seeds 17, 23 and 41.

Maximum 1,800 updates per seed, batch 16, AdamW LR 0.001, weight decay 0.01, gradient clip 1.0, deterministic CPU FP32, one torch thread. Use ordinary response-token SFT without an auxiliary numerical reward. Evaluate validation generation and response NLL at initialization and steps 300, 600, 1200 and 1800. Select by validation verified-prediction rate, then output validity, then lower NLL; break remaining ties in favor of the earlier checkpoint. Keep every evaluated checkpoint and failure. Save an emergency checkpoint if the training deadline is reached (20 minutes per seed); an incomplete run does not satisfy the training protocol.

Before holdout evaluation, freeze all three selected checkpoint hashes. No further training/config changes may use their test/OOD outcomes. If quality is inadequate, report the result and propose a new versioned experiment; do not keep tuning against this holdout. Implementation bugs may be corrected with retained failure records and a clearly disclosed rerun.

## Baselines and end-to-end evaluation

Compare a deterministic degree-2 interpolator with one candidate, the seed-matched initial native model, and the validation-selected trained native model. Each uses one candidate maximum, the same question/source/measurement cases, the verification-first rule Judge, eight-action upper budget and the same independent verifiers. The trained Judge is held out of this experiment to avoid confounding Researcher changes with the known calibration defect.

Evaluate all 64 in-family test and 64 OOD worlds through the actual Controller. Record per-case raw generated text, syntax/contract validity, unsupported evidence references, candidate error, VERIFIED/CONTRADICTED/UNKNOWN, tool count, wall time and full run/state/provenance artifacts. Report success per world and per family, with Wilson 95% intervals; do not treat the three seeds on shared worlds as 192 independent test tasks. Report seed dispersion separately. Include paired changed-success counts against each seed's initialization and explicit comparison with the deterministic reference.

The OOD goal is truthful handling of invalid or wrong candidates; the degree-2 output schema cannot represent a general cubic law. Any accidental agreement at one target remains only scoped agreement.

## Predeclared engineering acceptance gate

For **each** seed on the in-family holdout: contract-valid output ≥90%; at least 25% of worlds yield an independently VERIFIED prediction; verified success exceeds its random-initialization baseline by at least 10 percentage points. Across all evaluated runs: zero VERIFIED claims without actual passing provenance and measurement checks, and no fabricated alias accepted. The first three criteria are provisional engineering thresholds, not a scientific significance claim.

Report a gate miss as a miss. OOD accuracy has no success threshold; all failures and false confidence must remain visible. The experiment is complete when its registered measurements and failure analysis are recorded, whether or not the capability gate passes.

## Scope and next decision

Keep the default reference loop and phase-1 suite unchanged. The new contract is checkpoint-versioned and opt-in. Passing only establishes a learned structured component in a synthetic domain. It does not justify 100M/1B/7B training, a novel RLHF+RLVR+RLCD claim, or trusting Judge confidence. The physical VIT audit remains pending operator access and measurements.

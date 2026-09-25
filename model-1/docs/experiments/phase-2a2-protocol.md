# Phase 2A.2 — arithmetic curriculum and observation consistency

Preregistered 2026-09-25 before generating the new dataset or training. Version `aim-arithmetic-curriculum-v1`. Baseline evidence: [Phase 2A.1](../../reports/phase-2a1-implementation.md). This is a bounded experiment within the approved architecture; it does not approve larger models or a combined reward.

## Hypothesis and controls

**Hypothesis:** prerequisite arithmetic examples improve transfer to the full worked research task at fixed dense-compute budget. The preceding experiment improved two seeds but regressed on one, and some successful target predictions had incorrect process traces. It did not identify whether poor arithmetic fitting or poor transfer was responsible.

Compare `worked` (full worked research SFT only) with `curriculum` (the same task plus staged prerequisite SFT). Both use the existing worked output contract, random initialization, 228,096-parameter decoder, byte tokenizer, separate rule Judge and existing Controller/verifiers. No generated answer repair, constrained decoding, oracle at inference, or new RL reward. Teacher arithmetic appears only in explicitly labelled auxiliary training tasks.

## Fresh world boundary

Exclude all complete coefficient vectors used in every split of Phase 2A and Phase 2A.1 (1,408 vectors total). Compare actual vectors, not versioned hashes. The old linear pool is nearly exhausted, so expand c0,c1 to integers -19..19; keep c2 -3..3. This changes difficulty relative to historical studies; only the paired arms in this experiment estimate the curriculum effect.

Within each stratum sort remaining worlds by SHA-256 of this version and coefficient vector. Assign consecutive train/validation/test ranges: 128/16/16 linear and 384/48/48 quadratic. Choose 64 cubic OOD worlds similarly, with c3 in {-2,-1,1,2}, excluding historical cubic vectors. Observe x=0,1,2 and choose target from 3,4,5 using the new world hash. Keep cross-family observation ambiguities and report prompt overlap.

Trainer-readable metadata contains only versions, counts, group hashes, excluded-world hash and file hashes. Put new coefficient membership and the old exclusion registry in a separate audit file. The trainer must reject unrecognized metadata fields and never open holdout or audit files. All research rows preserve source spans, world identity and programmatic label origin. Auxiliary rows derive only from the corresponding train/validation worlds; no held-out world supplies auxiliary training examples.

## Fixed curriculum schedule

Both arms draw 16 research-world indices from the same seeded sampler every update. The curriculum arm replaces the first positions with auxiliary tasks derived from those sampled **training** worlds:

| Updates | Full research examples | Auxiliary examples | Auxiliary task |
|---|---:|---:|---|
| 1–600 | 8 | 8 | Signed subtraction: y1-y0 on odd updates, y2-y1 on even updates |
| 601–1200 | 12 | 4 | Second difference: y2-2*y1+y0 |
| 1201–1800 | 12 | 4 | Reconstruct c0,c1,c2 from supplied y0,d1,d2 |

Auxiliary outputs are strict JSON: `{"value":number}` for subtraction/differences, `{"coefficients":[[c0,c1,c2]]}` for reconstruction. Their prompts identify the operation. Full research prompts do not supply d1,d2; the model still generates all numbers itself.

Full research remains `{"d1":...,"d2":...,"coefficients":[[...]],"evidence":["E0"]}`. Ordinary response-token cross-entropy, including EOS, trains every task: L = -sum log p(target token | preceding tokens) / number of response tokens. No reward blending.

The worked arm receives 28,800 full research presentations per seed. The curriculum arm receives 19,200 full research, 4,800 subtraction, 2,400 second-difference and 2,400 reconstruction presentations. This is a comparison of whole training recipes at equal compute shape, not an isolated estimate controlling research-example exposure. Report task counts and response-token counts. A future exposure-matched third arm would be a separate experiment.

## Model, budget, selection and fitting diagnostics

Seeds 17,23,41; width96, layers2, heads4, KV2, FFN256, context256, CPU FP32, one thread. AdamW LR0.001, weight decay0.01, clip1, 1,800 updates, batch16. Pad both arms to256 input positions: 7,372,800 positions each. This matches dense tensor shapes, not measured FLOPs, wall time or supervised information quantity. Compare paired initial tensors, final sampler state and cumulative sampled-world digest. Preserve checkpoint/optimizer/RNG/curriculum accounting for exact continuation.

Validate at step0,300,600,1200,1800. Greedy generation cap128 for both arms. Select only by full-research validation prediction agreement, validity, lower response NLL, earliest tie. Freeze all six selected and initial checkpoint hashes before holdout scoring. Deadline1,200 seconds per seed; retain interrupted runs and emergency checkpoints. No adaptive extension or threshold changes.

At each validation boundary additionally generate on the first64 training research worlds by group order, and up to32 unique auxiliary prompts per task kind from train and validation. Deduplicate auxiliary prompts; exclude from validation diagnostics any prompt appearing anywhere in the training auxiliary pool, before selecting the first32 by prompt hash. Report eligible and excluded counts. These diagnostic scores never select checkpoints. Small diagnostic samples estimate neither complete training mastery nor broad arithmetic competence.

Report strict JSON validity and exact auxiliary answer accuracy, full-research training-versus-validation prediction agreement, coefficient/step accuracy and loss. An insufficient diagnostic pool is reported honestly rather than filled from training or test. The fixed schedule is not changed in response to diagnostics.

## Frozen loop evaluation and separate observation check

Evaluate the one-candidate deterministic reference, six initial and six selected models on64 test and64 OOD worlds through the existing eight-action Controller and two existing final verifiers: 1,664 runs. Preserve raw text, claims, scoped checks, provenance and failures. Report per-seed/per-family success, Wilson95 intervals, paired gains/losses, seed dispersion, process errors and cost. Seeds share worlds; do not pool them as independent tasks.

Add version1 of a **separate hypothesis observation-consistency checker**. It independently evaluates the generated polynomial with rational arithmetic at every cited observation, after validating source spans, topic and numeric schema. Bind results to the exact hypothesis plus evidence. Outcomes: PASS for all residuals <=1e-8; FAIL for a mismatch or broken provenance; UNKNOWN for missing/unsupported observations. It proves only finite observed-point agreement. It must never promote a target claim, replace the existing two checks, supply candidate coefficients, or label a global law proved.

Apply this checker after completed case runs and preserve a separate audit. Report target VERIFIED with observation inconsistency, and the diagnostic conjunction of target verification plus observation consistency. This conjunction is not a changed canonical success gate. Test counterexamples that match a future target but fail an observed point, observation-fit without future agreement, forged spans, missing evidence, incompatible topics, conflicting observations, numeric abuse, and hypothesis-hash changes. Read original memory without changing it.

## Decision rule and limitations

Each arm must satisfy each seed's original engineering gate: >=90% contract-valid, >=25% target-verified, >=10 percentage points over its initial model, plus zero unbacked VERIFIED claims. Curriculum advantage additionally requires mean gain >=5 percentage points over worked and no seed worse. Promotion requires both its arm gate and advantage gate. These provisional thresholds are not a statistical significance claim. Report a miss as a miss; keep the deterministic default unless the gate and appropriate review support a change.

No OOD success threshold. Training/validation diagnostics and the new consistency audit explain results without tuning on test. Historical evaluations remain unchanged and exposed. A positive outcome would establish a tiny synthetic recipe improvement, not general research intelligence, a validated RLHF+RLVR+RLCD combination, or VIT cluster feasibility. Keep the current model scale and separate training stages.

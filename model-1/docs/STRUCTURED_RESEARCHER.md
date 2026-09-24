# Structured Researcher experiment — Phase 2A

This phase trains the native decoder on research-shaped supervision and evaluates it through the existing Controller. It is the next step after the phase-1 arithmetic checkpoint failed structured generation. The default reference Researcher, component architecture and original eight-case regression suite remain unchanged.

The [registered protocol](experiments/phase-2a-protocol.md) defines the hypothesis, data, model, budgets, baselines, selection rule and numerical gates before holdout evaluation. Current measured results belong in the phase-2 report, not in this protocol.

## What changed

- `research_format.py`: versioned input/output schema; local evidence aliases resolve to actual source-span IDs; malformed JSON, duplicate keys, nonfinite coefficients and invented/missing citations are rejected.
- `research_data.py`: deterministic world-level partitions and source-backed supervision. Training and validation are exported separately from test/OOD data.
- `researcher.py`: the existing transformer adapter dispatches to the new schema only when its checkpoint declares the matching contract version. Legacy phase-1 checkpoints retain their old contract. No reference fallback is used.
- `controller.py`: records the prompt, raw generated text, evidence mapping and parse outcome, including rejected outputs, in the event history.
- `research_train.py`: native random-initialization SFT, three independent seeds, bounded updates, evaluated checkpoints, validation-only selection and a frozen checkpoint manifest.
- `research_eval.py`: one-candidate reference versus matched random/trained checkpoints, actual tool/verifier execution, world-level success, descriptive Wilson intervals, paired success changes, per-seed gates and OOD diagnostics.

No new pretrained weights, tokenizer, reward mixture, external API or model architecture is introduced. The experimental decoder has 228,096 parameters; its width/FFN increase is a bounded configuration within the existing dense architecture and 2M microbenchmark guard.

## Dataset and provenance

The generator writes `train-validation.json`, `holdout.json`, `split-manifest.json` and a source-memory store. Every row preserves the world group, family, split, generator, programmatic label origin, rights declaration, source hashes, alias mapping, prompt and complete local case. Train/validation rows additionally carry a target response. The hidden world coefficients and target measurement are retained for audit and scoring but are never passed to the neural prompt.

There are 512 training worlds, 64 validation worlds, 64 in-family test worlds and 64 OOD cubic worlds. Each in-family split has 25% linear/constant and 75% quadratic worlds. Splits are disjoint by complete coefficient world. This is a tiny finite synthetic universe, not an estimate of general mathematical capability.

With only three observations, a cubic world can be observationally indistinguishable from a quadratic world in the training set. The split manifest explicitly counts exact OOD/training prompt overlap. Such ambiguity is retained and disclosed; it is not removed after seeing performance. It reinforces why agreement at one measurement cannot prove a general law.

Each generated training case currently has one observation document, aliased E0. Passing citation syntax on these fixtures does not establish sophisticated multi-document attribution or entailment. Unknown aliases are rejected by the parser, and original spans are checked independently by the Controller.

## Reproduce from the committed source

Run from `model-1/` with the pinned local environment:

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m aim.research_data
.venv/bin/python -m aim.research_train \
  --data runs/<research-dataset-run>/train-validation.json
.venv/bin/python -m aim.research_eval \
  --selection runs/<research-training-run>/selected-models.json \
  --holdout runs/<research-dataset-run>/holdout.json
```

Replace the paths with the actual directories printed by each stage. Training prints validation summaries at the declared checkpoints. The evaluation prints every backend/split summary and writes full case records. A capability gate miss is a valid experimental outcome and does not change process exit status to an infrastructure failure; inspect `gate_passed` in the metrics.

To use a selected native Researcher in the ordinary loop:

```sh
.venv/bin/python -m aim loop --researcher-checkpoint <selected-checkpoint.pt>
```

Choose the path from `selected-models.json`, keep its integrity sidecar beside it, and inspect UNKNOWN/CONTRADICTED outcomes as well as VERIFIED. The deterministic default remains available as the baseline.

## Training and checkpoint semantics

Each seed starts from its own random weights and uses ordinary response-token cross-entropy, including EOS. There is no coefficient reward, candidate-answer oracle, grammar-constrained decoding or selection among generated hypotheses. Generation is greedy and bounded by context and maximum new tokens. The three coefficient values and citation syntax must be produced by the decoder.

Validation response NLL is measured in bounded batches. Generated candidates are checked against the known validation measurement using exact rational arithmetic; this is called **prediction agreement**, not a scientific verification claim. Final test/OOD measurements use actual Controller/tool/Verifier runs and their scoped VERIFIED statuses.

Checkpoint selection uses validation agreement, then contract validity, then lower response NLL, then earliest step. All seeds' chosen hashes are frozen before invoking the holdout evaluator. Both initial and selected checkpoint hashes are rechecked there. One shared test set is used across seeds; do not pool them as independent worlds.

Evaluated checkpoints preserve weights, optimizer, sampler RNG, torch RNG, step, seed, tokenizer, dataset identity and checkpoint-parent hash. The research-specific trainer currently runs a complete bounded seed from scratch; it does **not** yet expose a tested `--resume` interface. The older arithmetic trainer's verified resume support must not be attributed to this new trainer. Emergency deadline checkpoints retain state for debugging; add and test a compatible continuation path before using this trainer for long runs.

## Failure interpretation

- Invalid JSON/schema/citations: zero valid hypothesis; Controller returns unresolved; raw generation remains in the trace.
- Valid but incorrect numbers: independent measurement fails and the claim is CONTRADICTED.
- Matching measurement: VERIFIED only at the specific target/tolerance, with checked provenance.
- Different-family world: report the result even when syntax looks correct. An unsupported degree-2 law does not become a general cubic explanation.
- Deadline, corrupted checkpoint, changed training data or unexpected code error: failed run retained; no score manufactured.

The rule Judge is used throughout this comparison to isolate the Researcher change from the already observed calibrated-Judge defect. Preference/RLVR/Judge training are still separate mechanisms; this phase does not demonstrate their combined benefit.

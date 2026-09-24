# Plain versus worked supervision — reproduction guide

The [registered Phase 2A.1 protocol](experiments/phase-2a1-protocol.md) defines this experiment before its holdout is used. It follows Phase 2A's numerical failure and the addition of tested Researcher checkpoint continuation. It changes supervision and the experimental output contract, preserving the approved component architecture and model size.

## Run the complete experiment

From `model-1/`, using the recorded Python/torch environment:

```sh
.venv/bin/python -m aim.comparison_reproduce
```

This runs all tests with a 120-second test deadline and requires zero skips, generates the fresh dataset, trains plain/worked arms for all three seeds, freezes all checkpoint selections, and only then evaluates holdout cases through the Controller. Stage pointers are written immediately after successful stages. A failed run retains its log and artifacts. A model capability gate miss is a completed measurement, not an infrastructure error.

The default configuration is `configs/finite-difference-comparison.json`. The bounded experiment uses 228,096 parameters, six training runs and 1,800 updates per run. It never downloads pretrained assets or sends data to an external model. Reserve local disk for every checkpoint, source archive, case trace and memory database; these are intentionally retained.

Individual stages:

```sh
.venv/bin/python -m aim.comparison_data
.venv/bin/python -m aim.comparison_train \
  --data runs/<comparison-dataset>/train-validation.json
.venv/bin/python -m aim.comparison_eval \
  --selection runs/<comparison-training>/selected-models.json \
  --holdout runs/<comparison-dataset>/holdout.json
```

Use actual printed run paths. Do not retrain against an inspected holdout and describe it as a new blind comparison. A different configuration is a different experiment, even if the command accepts it for miniature integration checks.

## What each implementation owns

| File | Responsibility |
|---|---|
| `finite_difference.py` | Versioned worked schema, programmatic labels, independent observational arithmetic diagnostics |
| `comparison_data.py` | Exclude all Phase 2A coefficient vectors; create shared fresh worlds and source records |
| `comparison_train.py` | Transform training responses per arm, invoke native trainer, validate equal initial tensors and processed positions, freeze selections |
| `research_train.py` | SFT updates, optimizer/sampler state, budget accounting, validation-only selection and resumable seed training |
| `comparison_eval.py` | Actual Controller runs, paired metrics, process failures, descriptive uncertainty and registered gates |
| `comparison_reproduce.py` | Retained preflight log and stage-by-stage reproduction record |

The original Phase 2A generator remains available and deterministic. Its complete coefficient vectors form the exclusion registry for this experiment. The new manifest includes that registry and its hash, all new world vectors, counts, and data hashes. This prevents a version-prefix change from hiding old-world reuse.

## Understand the compute comparison

Both arms draw the same worlds in the same order for each seed, start with equal tensors, and execute the same batch size, number of updates and dense tensor shapes. `sequence_scores(..., pad_to=256)` places padding after the supervised sequence. Padding has zero loss weight; earlier causal positions cannot attend to it. Tests compare response-token counts and scores with and without padding.

Each arm/seed processes exactly `1800 × 16 × 256 = 7,372,800` training input positions. This matches a dense-compute proxy, not measured hardware FLOPs. Worked responses contain more supervised tokens; their quantity is reported, not claimed equal. Generation stops at EOS or 128 new tokens for both arms. Output UTF-8 byte counts are logged; for malformed decoded text these are not an exact count of generated token IDs. Inference latency includes parsing, tools, verification and trace persistence.

Training metrics record cumulative processed positions and supervised tokens. On resume from a legacy checkpoint lacking those counters, `accounting_start_step` states where accounting begins; missing historical cost is never reported as a complete total. Optimizer state, sampling and selection history are still restored. The comparison reproduction command starts fresh paired runs; it does not automatically assemble an interrupted six-run comparison from separate continuations.

## Interpret verification correctly

The worked schema requires finite `d1` and `d2`, one three-coefficient candidate, and supported evidence aliases. Parsing accepts structurally legal but incorrect arithmetic. The existing Controller performs its usual provenance and new-measurement checks. Process diagnostics separately compare the generated numbers to exact finite differences of the supplied observations.

Therefore a final prediction can be VERIFIED while a worked step is wrong. The report explicitly counts this. Never describe such an output as a verified proof or a fully checked reasoning trace. Conversely, valid syntax and correct finite differences need not predict a cubic world's next measurement. A quadratic output cannot represent the full cubic law.

Coefficient diagnostics measure agreement with the unique degree-at-most-two interpolant of the three observations. This equals the hidden coefficients for in-family fixtures. For cubic OOD, it is only an observation-consistency diagnostic; it is not cubic parameter recovery.

This is programmatic SFT with step diagnostics. Human preference learning, reward-model training, RLVR policy optimization and Judge calibration remain separate future comparisons.

## Artifact navigation

- Reproduction root: `manifest.json`, `tests.log`, `test-result.json`, stage pointers, `summary.json`, completion/failure.
- Dataset child: source memory, training/validation file, separate holdout, exclusion/split manifest.
- Training child: transformed data for each arm, six seed runs, frozen `selected-models.json`.
- Seed run: configuration/environment, evaluated checkpoint files plus integrity sidecars, validation generations, metrics, parent lineage, event log and optional emergency checkpoint.
- Evaluation child: accepted frozen selection, all case traces and state, per-case comparison/step diagnostics, aggregate metrics and gate outcomes.

Keep ancestor run directories and integrity sidecars. Historical selection paths currently use absolute local paths; portable reports preserve hashes and repository-relative artifact references. Cross-machine artifact relocation remains an engineering limitation. The VIT fleet has not been measured by this local experiment.

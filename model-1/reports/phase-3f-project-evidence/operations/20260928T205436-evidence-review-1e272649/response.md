# AIM local evidence review

## Question

```text
calibration OOD uncertainty
```

Source excerpts below are **UNVERIFIED assertions**. A PASS applies only to attribution.

## Excerpt 1 — UNVERIFIED

```text
ion log loss, and use q=σ(gφ/T). Neither test nor OOD labels select T. This is a deliberately small grid-search implementation of temperature scaling, whose empirical calibration role is discussed in [Guo et al., On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).

The experiment demonstrates that temperature fitting on familiar data can worsen shifted-family confidence. It supplies no guarantee of calibration on scientific research claims. The Judge may ABSTAIN below a configured threshold; the default threshold 0 requests verification for all candidates. Runtime verification remains mandatory even for high-confidence predictions.

This is **RLCD-style decision calibration by supervised proper scoring**. It is not reinforcement learning on a sequential decision MDP, nor a reproduction of proprietary RLCD training. To move to decision-policy RL, first define actions, observable state, time-dependent outcomes, costs, delayed feedback and off-policy evaluation. Keep that proposed method
```

```text
{
  "uri": "aim:docs/TRAINING.md",
  "version": "sha256:2e3c8d3b769f93a79ad185bb48d63247be418aaa981fa69aee5c4e83546c7bcc",
  "title": "Model-1 training specification",
  "source_id": "lib-src-490dcc1789716918e289537e34b05ecebe53eab1cd208b34fa10a4ca757ed8cc",
  "chunk_id": "lib-chunk-b0a7ba5d803f90757a63b299628fa1cd8d679eb8b3373949d5336416026ff687",
  "start": 9856,
  "end": 10880,
  "evidence_id": "ev-0fa1a5a616a613ecffac3e70"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 2 — UNVERIFIED

```text
 gate compares rich/log/temperature models with the
base-rate and verify-all baselines, including an OOD utility and error constraint.
Passing it would justify further study, not default replacement. Learned
Researcher candidates, language evidence, robust calibration on new domains,
human feedback and sequential decisions require separate experiments.

## Modules

| Module | Purpose |
|---|---|
| `judge_shift_features.py` | Restricted input contract and observed-only features |
| `judge_shift_data.py` | Grouped fresh worlds, Memory evidence, verifier labels |
| `judge_shift_train.py` | Separate proper-score fits, calibration freeze, runtime adapter |
| `judge_shift_eval.py` | Frozen scoring, policies, slices and grouped intervals |
| `judge_shift_reproduce.py` | Tests and retained full experiment |
| `tests/test_judge_shift.py` | Input leakage, data/label invariants and actual Controller cases |

The [protocol](experiments/phase-2b-protocol.md#primary-evidence-for-methods)
links the primary scoring-rule and 
```

```text
{
  "uri": "aim:docs/JUDGE_SHIFT.md",
  "version": "sha256:ff0670e4897d16fc22a7c724a4098b2fa977961854943e1730bb2192e4fd3124",
  "title": "Separate Judge forecasts and decisions under shift",
  "source_id": "lib-src-d4065518beea83b70aeda75874a6f0444eff1bf840483244859af2c52e2f69af",
  "chunk_id": "lib-chunk-db815b906496eda36d7abc0c7ddc557471705695a4ffc90aa2add5e55e4a5750",
  "start": 6272,
  "end": 7296,
  "evidence_id": "ev-49217e7f489655430c8f44ca"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 3 — UNVERIFIED

```text
# Language fitting diagnostics

Phase 3E extends the versioned language path with fixed train/validation probes. It does not introduce a new model architecture or merge training objectives. Read the [predeclared protocol](experiments/phase-3e-protocol.md) for the exact experiment and interpretation limits.

**Executed evidence:** [221 tests passed; eight arms, 32 measurements and 40 Controller replays](../reports/phase-3e-implementation.md). Structure improved after 384 updates, but each arm had zero validation checker passes. The standalone diagnostic command exactly reproduced one recorded measurement with one compute thread requested through `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. Neither the study nor the CLI check promotes a learned model.

## Interfaces

`aim.language_diagnostics.freeze_probes` selects the first requested rows from both prepared splits, binds the full prepared dataset hash, and freezes all selected rows. Only structured numerical and symbolic SFT tasks are supported. Test/OOD rows 
```

```text
{
  "uri": "aim:docs/LANGUAGE_FITTING.md",
  "version": "sha256:8db39a232634a32c130df2cbab8379da9bd1d95a1033ec533d29b55ee1a9a4fb",
  "title": "Language fitting diagnostics",
  "source_id": "lib-src-344676b01d020c3e0f14ee58fb3e7f9b68bf4daa4cd48e97816c932e9ef1488e",
  "chunk_id": "lib-chunk-3c12cf9eba1eeac5c8a31e12da6abc57552a284538de276949583c7e3375a5cf",
  "start": 0,
  "end": 1024,
  "evidence_id": "ev-2f83520fce214474541aa44e"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 4 — UNVERIFIED

````text
ared dataset hash, and freezes all selected rows. Only structured numerical and symbolic SFT tasks are supported. Test/OOD rows and changed manifests reject. Checkpoint evaluation also reconstructs the expected probes from the checkpoint's adjacent `dataset.json`; a matching declared dataset hash alone is insufficient.

`measure` runs teacher-forced scoring and strict greedy generation without changing model weights, mode or CPU RNG. `measure_checkpoint` additionally binds task, tokenizer, model and dataset identities. Keep a checkpoint with its `.sha256.json` sidecar and `dataset.json` when using this command:

```sh
.venv/bin/python -m aim.language_diagnostics \
  --checkpoint <training-run>/checkpoint.pt \
  --probes <study-run>/structured-probes.json \
  --max-new-tokens 128
```

The command creates an immutable tracked run containing `diagnostics.json`; failures remain failed tracked runs. It does not modify the checkpoint or research memory.

## Measurements and boundaries

Each probe records gold-prefi
````

```text
{
  "uri": "aim:docs/LANGUAGE_FITTING.md",
  "version": "sha256:8db39a232634a32c130df2cbab8379da9bd1d95a1033ec533d29b55ee1a9a4fb",
  "title": "Language fitting diagnostics",
  "source_id": "lib-src-344676b01d020c3e0f14ee58fb3e7f9b68bf4daa4cd48e97816c932e9ef1488e",
  "chunk_id": "lib-chunk-3c8b580c438ff1a48374ea93dbed004ac6a93989d61e595a684f6fa1ac77cdc7",
  "start": 896,
  "end": 1920,
  "evidence_id": "ev-6d5e48ffb553a7061ab8adf3"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 5 — UNVERIFIED

```text
. No PPO clipping, GRPO, learned critic, process reward, long-horizon credit assignment or free-form rollout engine is claimed.

The two branches have different total prior training budgets. Their single-seed results are diagnostics, not a controlled proof that DPO helps or harms RLVR. A future ablation must match total examples, updates, verifier calls and compute across seeds.

## Objective 4: separate calibrated Judge

The defined event z=1 is: “this candidate prediction matches the next synthetic measurement within absolute tolerance 10⁻⁸.” For pre-measurement features f and separate weights φ, qφ=σ(gφ(f)). Implemented proper scoring options are:

\[
L_{log}=-[z\log q_\phi+(1-z)\log(1-q_\phi)],\qquad
L_{Brier}=(q_\phi-z)^2.
\]

The default trains log loss for 150 steps, batch 32, learning rate 0.01. Afterwards choose T from the declared grid {0.5,0.75,1,1.25,1.5,2,3,5} by validation log loss, and use q=σ(gφ/T). Neither test nor OOD labels select T. This is a deliberately small grid-search implementation o
```

```text
{
  "uri": "aim:docs/TRAINING.md",
  "version": "sha256:2e3c8d3b769f93a79ad185bb48d63247be418aaa981fa69aee5c4e83546c7bcc",
  "title": "Model-1 training specification",
  "source_id": "lib-src-490dcc1789716918e289537e34b05ecebe53eab1cd208b34fa10a4ca757ed8cc",
  "chunk_id": "lib-chunk-8897494e7f4568b208828e888719c81ddc5a21f031d3ca1290053b88366f1967",
  "start": 8960,
  "end": 9984,
  "evidence_id": "ev-1a0aa12ce84831211b147879"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 6 — UNVERIFIED

```text
cast is attached.

Five features preserve the prior Judge baseline: candidate coefficient count,
observation count, mean normalized residual, target distance, and normalized
candidate prediction magnitude. The seven-feature version additionally receives
an exact finite-fit indicator and maximum normalized residual. The indicator is
a numerical feature; it does not independently approve source provenance.

Source-backed generated observations and exact rational synthetic measurements
are retained in the dataset Memory. `MeasurementVerifier` supplies labels. The
training file contains only training/calibration records. A metadata allowlist
checks the training hash and carries a holdout hash without holdout labels.
Hidden coefficients and measurement/check details live in a separate audit file.
Both hidden variants and both candidates for a base group stay in one split.

The loader recomputes all features from the permitted input and rejects altered
values. Row IDs, families, source records and groups are attrib
```

```text
{
  "uri": "aim:docs/JUDGE_SHIFT.md",
  "version": "sha256:ff0670e4897d16fc22a7c724a4098b2fa977961854943e1730bb2192e4fd3124",
  "title": "Separate Judge forecasts and decisions under shift",
  "source_id": "lib-src-d4065518beea83b70aeda75874a6f0444eff1bf840483244859af2c52e2f69af",
  "chunk_id": "lib-chunk-0c4a81fa9be58f22e7dd9314d47ed3fe8c5e12ecfa0010200b43209660156c5c",
  "start": 1792,
  "end": 2816,
  "evidence_id": "ev-498ab1649d8801cdb2a10fce"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Limits

- Source statements are UNVERIFIED. Exact attribution does not establish factual truth.
- Lexical retrieval is incomplete and may miss paraphrases or split terms at chunk boundaries.
- Versions and mirrors of a source are not independent corroboration.
- The reference Researcher extracts text; no learned synthesis or hypothesis testing is performed.
- The reference Judge has no learned or calibrated probability.
- Sources are active at the recorded snapshot; consult the current library for later retirement.

## Snapshot

```text
{"library_id": "84e25418110b4debb3de0769e0321d32", "sequence": 6, "sha256": "b0c06e1e19c172c562ce7565e621645f42bf594b2428ef1c68a48b44a75b1ca5"}
```

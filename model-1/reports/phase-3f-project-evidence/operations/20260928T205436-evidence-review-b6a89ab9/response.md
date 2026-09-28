# AIM local evidence review

## Question

```text
RLHF preference optimization RLVR reward calibrated Judge
```

Source excerpts below are **UNVERIFIED assertions**. A PASS applies only to attribution.

## Excerpt 1 — UNVERIFIED

```text
 Cross-device and cross-library-version bitwise continuation is not promised.

Nonfinite loss/gradient, incompatible checkpoint, invalid data, excessive allocation or illegal resume fails with retained traceback and configuration. Unknown research outcomes are not discarded or turned into incorrect numeric labels.

## Integration policy and remaining experiments

At runtime the components exchange structured state and evidence; training rewards are not added together. The Controller applies verification constraints after the Judge's forecast. This separation makes SFT, preference learning, verifiable outcomes and calibration independently measurable.

Before proposing joint training, pre-register comparisons: SFT-only; SFT+DPO; SFT+RLVR; SFT+DPO+RLVR; each with rule Judge versus separately calibrated Judge. Evaluate budget, abstention, domain shift, fabricated citations and verifier exploitation. Any future combined scalar objective or constrained optimization must state units, coefficients, constraint thresh
```

```text
{
  "uri": "aim:docs/TRAINING.md",
  "version": "sha256:2e3c8d3b769f93a79ad185bb48d63247be418aaa981fa69aee5c4e83546c7bcc",
  "title": "Model-1 training specification",
  "source_id": "lib-src-490dcc1789716918e289537e34b05ecebe53eab1cd208b34fa10a4ca757ed8cc",
  "chunk_id": "lib-chunk-26c07443faba46270e7e7ee03d215cd10a8e2464515f4bdd731762c9e4b07c9d",
  "start": 11648,
  "end": 12672,
  "evidence_id": "ev-320f9d9b223567613f938de7"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 2 — UNVERIFIED

```text
exploitation. Any future combined scalar objective or constrained optimization must state units, coefficients, constraint thresholds, failure semantics and reference distributions, and compare against those separated baselines.

## Symbolic stages

`task: symbolic` selects separately versioned binomial data for the SFT, preference and RLVR trainer. Symbolic and legacy checkpoints cannot be interchanged silently. The separate `symbolic-judge-train` entry point fits a symbolic checker-PASS forecast, with its own checkpoint kind, features, calibration split and tested exact resume. See [objectives, data and commands](SYMBOLIC.md).

Programmatic preferences do not constitute collected human feedback. Symbolic UNKNOWN is a zero reward for the precisely defined checker-PASS target, not a false theorem label. The [first build](../reports/phase-2c-implementation.md) saved all four models but achieved zero verified free-generation test answers; finite-candidate RLVR scores were not a reliable indicator of generation q
```

```text
{
  "uri": "aim:docs/TRAINING.md",
  "version": "sha256:2e3c8d3b769f93a79ad185bb48d63247be418aaa981fa69aee5c4e83546c7bcc",
  "title": "Model-1 training specification",
  "source_id": "lib-src-490dcc1789716918e289537e34b05ecebe53eab1cd208b34fa10a4ca757ed8cc",
  "chunk_id": "lib-chunk-ba4bad7965f04b94caeff70f180a946f3f224a64181b33544ed6094d29738b6f",
  "start": 12544,
  "end": 13568,
  "evidence_id": "ev-171d456f371ab85be0a71137"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 3 — UNVERIFIED

```text
kpoints

The native random-init causal decoder uses CPU FP32 and AdamW. With target mask m, the objective is:

`L = - sum_{b,t} m[b,t] log p_theta(y[b,t] | x[b,0:t]) / sum_{b,t} m[b,t]`.

This is ordinary next-token pretraining; it supplies no RLHF/RLVR/Judge reward. Nonfinite loss or gradients fail the run. Training is bounded to 1,000 steps, batch at most 16, context at most 512 and 2M parameters. The smoke config is substantially smaller. These are harness limits, not scientific scaling recommendations.

Periodic and final checkpoints record weights, optimizer, Torch RNG, global step, scored-token count, complete stream cursor, corpus identity, tokenizer specification, stable configuration and parent checksum. Resume may increase the total step target but rejects other configuration, data or tokenizer changes. Existing checkpoints are never overwritten. Publication uses an exclusive weight file followed by its hash sidecar; a partial file or missing sidecar cannot load. Prior completed checkpoints remain u
```

```text
{
  "uri": "aim:docs/PRETRAINING.md",
  "version": "sha256:0f46c424e763e55d05ab8dae57f1c24c5e323f3d221ea73c6fd0553e6546e052",
  "title": "Phase 3A — From-scratch corpus and pretraining infrastructure",
  "source_id": "lib-src-68b806a108ce1d20d297100c6af1804b8d196a44819a7f5e5376550b50401cca",
  "chunk_id": "lib-chunk-c236a7bf4a98877d074144ff705fd546431aa3fe21920420293ef1ec478ecff0",
  "start": 6272,
  "end": 7296,
  "evidence_id": "ev-38eaaf67ae8317e3dbb9a57f"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 4 — UNVERIFIED

````text
# Separate Judge forecasts and decisions under shift

Phase 2B implements the [registered protocol](experiments/phase-2b-protocol.md).
It leaves the default Controller/Judge, earlier checkpoints and canonical suites
unchanged. The opt-in Judge is a feature MLP trained from random initialization,
not a language Judge or a new Researcher. No SFT/preference/RLVR rewards are mixed.

## Reproduce

From `model-1/`, using the recorded environment:

```sh
.venv/bin/python -m aim.judge_shift_reproduce
```

The driver runs the full unit suite, rejects skips/failures, creates fresh grouped
worlds, trains twelve small Judges, freezes checkpoint hashes and temperatures,
then loads holdout and writes every forecast. Runs are retained with source
snapshots, config/environment/commit, dataset hashes, optimizer/RNG states and
explicit completion/failure. Completion does not imply an evidence gate passed.
The fixed new-study trainer does not yet support checkpoint resume.

Read root `summary.json`, `tests.log`, `dataset-stage.
````

```text
{
  "uri": "aim:docs/JUDGE_SHIFT.md",
  "version": "sha256:ff0670e4897d16fc22a7c724a4098b2fa977961854943e1730bb2192e4fd3124",
  "title": "Separate Judge forecasts and decisions under shift",
  "source_id": "lib-src-d4065518beea83b70aeda75874a6f0444eff1bf840483244859af2c52e2f69af",
  "chunk_id": "lib-chunk-123f3fb6ee1cb601042610b9c208227ac1ff12df0058c1b7cd26f80b99e23461",
  "start": 0,
  "end": 1024,
  "evidence_id": "ev-b24eea21c038aba29a301754"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 5 — UNVERIFIED

````text
me adapter's
threshold is an experimental heuristic in that shared-tool environment.

## Integration

The experimental adapter implements the existing Judge interface:

```python
from aim.judge_shift_train import ShiftJudge
from aim.controller import Controller
from aim.tracking import Run
from pathlib import Path

judge = ShiftJudge(checkpoint_path, calibrated=True, cost=0.5)
with Run(Path("runs"), "shift-judge-loop", {"cost": 0.5}, inputs=[checkpoint_path]) as run:
    state = Controller(judge=judge).run(case, run)
```

Use a real checkpoint from the frozen manifest and a valid existing case. The
legacy `--judge` CLI loader remains specific to the older five-feature checkpoint
format. Unit integration executes the new adapter through the actual Controller
with both verification and abstention, plus conflicting evidence. Confidence
does not bypass independent verifiers or promote an unmeasured claim.

## Shift and interpretation

Three observed points cannot distinguish q from q+k*x*(x-1)*(x-2), or the
regis
````

```text
{
  "uri": "aim:docs/JUDGE_SHIFT.md",
  "version": "sha256:ff0670e4897d16fc22a7c724a4098b2fa977961854943e1730bb2192e4fd3124",
  "title": "Separate Judge forecasts and decisions under shift",
  "source_id": "lib-src-d4065518beea83b70aeda75874a6f0444eff1bf840483244859af2c52e2f69af",
  "chunk_id": "lib-chunk-91716d4469b3ce21df347d33cf4a0d5ac8ed1d948fb3e40080b338844fb6fd99",
  "start": 4480,
  "end": 5504,
  "evidence_id": "ev-506d75f021e31f58a9801075"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 6 — UNVERIFIED

```text
# Phase 3A — From-scratch corpus and pretraining infrastructure

## What this build decides

Implement local corpus intake, reversible tokenizer candidates, document-at-a-time next-token training and exact CPU continuation before a larger model run. Keep the 2M-parameter engineering allocation guard. Byte tokenization remains the compatibility baseline; the new byte-pair tokenizer is an explicitly bounded experiment. No final tokenizer, corpus mixture or parameter count is selected here.

This supports the approved native Researcher architecture. It does not replace the Controller, Judge, verifier or memory components, and does not combine their training objectives.

## Corpus intake contract

`aim-corpus-manifest-v1` contains a version and a document list. Each document requires:

- Unique ID, relative local path, SHA-256, declared split and group.
- Source URI and version, plus a content category: prose, code, math or Unicode.
- Rights fields: license string, explicit Boolean `training_allowed: true`, basis
```

```text
{
  "uri": "aim:docs/PRETRAINING.md",
  "version": "sha256:0f46c424e763e55d05ab8dae57f1c24c5e323f3d221ea73c6fd0553e6546e052",
  "title": "Phase 3A — From-scratch corpus and pretraining infrastructure",
  "source_id": "lib-src-68b806a108ce1d20d297100c6af1804b8d196a44819a7f5e5376550b50401cca",
  "chunk_id": "lib-chunk-624761410a4c70bae7049d5737e69b15e74f873f97d51a558e22c039d0dac4ee",
  "start": 0,
  "end": 1024,
  "evidence_id": "ev-50f946d9db753037a7aa249f"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 7 — UNVERIFIED

```text
e floor assumes weights 4N + gradients 4N + two optimizer moments 8N = 16N bytes. It excludes activations, attention buffers, allocator/workspace, reference models, runtime, data, checkpoints and OS needs. An FP32 frozen DPO/RLVR reference adds approximately 4N bytes. Therefore a 32 GB desktop's ability to hold weights does not establish the feasibility or speed of a full training stage.

The smoke training harness caps model allocation at 2M parameters. A reviewed scale-trial runner must replace that guard deliberately, with memory estimation and an explicit time/token budget; do not merely edit the integer and launch 1B.

Sharding or pipeline/model parallelism would be necessary for some candidates on 32 GB hosts, but introduce substantial communication and failure-recovery work. FSDP/ZeRO, CPU offload, mixed precision and MoE are controlled future experiments, not implemented capabilities. Current GQA repeats KV tensors, preserving the small model's architecture but not optimizing the CPU memory footprint 
```

```text
{
  "uri": "aim:docs/HARDWARE_AND_CLUSTER.md",
  "version": "sha256:941ac209ea90192bd70128721971e7db1355825059917acee99f588509be7771",
  "title": "Hardware and cluster engineering protocol",
  "source_id": "lib-src-ed4b2b930c4c34e24e39b4ba4e4b6210b360f615523827299d03bbabcbc0e22c",
  "chunk_id": "lib-chunk-134ac2191003f0d11e03f0502b27af6981b6ce60e58900274d0a1b4e50b4738a",
  "start": 7168,
  "end": 8192,
  "evidence_id": "ev-87842cd5d2ba1963f633e01e"
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

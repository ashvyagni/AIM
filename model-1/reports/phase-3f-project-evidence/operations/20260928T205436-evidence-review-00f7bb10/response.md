# AIM local evidence review

## Question

```text
free generation validation checker fitting
```

Source excerpts below are **UNVERIFIED assertions**. A PASS applies only to attribution.

## Excerpt 1 — UNVERIFIED

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

## Excerpt 2 — UNVERIFIED

````text
alid polynomial expression can pass the checker without matching reference tokens.

The numerical diagnostic uses exact rational residuals against prompt observations and the fixture target, with tolerance 1e-8. Both must agree for its diagnostic PASS. Its target is evaluator-only fixture information; only prompt text reaches generation. This check has no claim-changing authority and does not prove source provenance or global correctness. The real numerical Controller checks its own target/provenance scope separately. The symbolic diagnostic uses the existing bounded Q[x] identity checker; UNKNOWN, FAIL and invalid output are distinct.

## Study runner

```sh
.venv/bin/python -m aim.language_fit \
  --config configs/language-fitting.json \
  --regression --export reports/<new-directory>
```

The fixed plan uses two domains, byte/BPE32 tokenizers, two seeds, a random-init baseline and three SFT budgets. Checkpoint continuation retains the original frozen reference. The runner writes prepared data, frozen probe
````

```text
{
  "uri": "aim:docs/LANGUAGE_FITTING.md",
  "version": "sha256:8db39a232634a32c130df2cbab8379da9bd1d95a1033ec533d29b55ee1a9a4fb",
  "title": "Language fitting diagnostics",
  "source_id": "lib-src-344676b01d020c3e0f14ee58fb3e7f9b68bf4daa4cd48e97816c932e9ef1488e",
  "chunk_id": "lib-chunk-e3dd3c57e889ecf970629d8d3167f09e1dcd78ff991e4d2ad1049a17a2b83e53",
  "start": 2688,
  "end": 3712,
  "evidence_id": "ev-c891051b552e6bc3e27ea824"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 3 — UNVERIFIED

```text
# Model-1 training specification

**Latest fitting diagnostics:** Phase 3E measured random-init and fixed SFT budgets under the [registered fitting protocol](experiments/phase-3e-protocol.md). [All eight arms completed](../reports/phase-3e-implementation.md); output contracts improved, but validation checker passes remained zero. Use the [diagnostic commands](LANGUAGE_FITTING.md) to distinguish format fitting from scoped correctness. This is a separate SFT experiment, not evidence that preference/RLVR/Judge objectives jointly improve research quality.

**Audited Phase 3D extension:** [tokenizer-compatible post-training](TOKENIZED_POSTTRAINING.md) introduces a separate versioned byte/BPE path using the objectives below. Its [executed audit](../reports/phase-3d-audit.md) passed 208 tests and exact continuation for all six tokenizer/stage combinations. Later preference/RLVR updates did not improve arithmetic response NLL in this miniature run. This document's earlier results/configuration describe the legacy pat
```

```text
{
  "uri": "aim:docs/TRAINING.md",
  "version": "sha256:2e3c8d3b769f93a79ad185bb48d63247be418aaa981fa69aee5c4e83546c7bcc",
  "title": "Model-1 training specification",
  "source_id": "lib-src-490dcc1789716918e289537e34b05ecebe53eab1cd208b34fa10a4ca757ed8cc",
  "chunk_id": "lib-chunk-fc0eebdf3558aff116d5b619f3cbf496ad1cf7e371147862a7b09a3e2225597f",
  "start": 0,
  "end": 1024,
  "evidence_id": "ev-19e0d9545677d16b0b069779"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 4 — UNVERIFIED

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

## Excerpt 5 — UNVERIFIED

```text
l paths, rights and review declarations remain in the local index. A failed intake keeps its run status, accepted-object history and traceback; no completed index is published.

Loaders verify index identity and recheck each object's hash and metadata when loading it. Hashes are corruption checks, not signatures against a privileged actor who rewrites both content and metadata. Do not publish external/private corpus objects or manifests through the engineering fixture exporter.

## Tokenizer candidates

The existing UTF-8 byte tokenizer now lives in a dependency-free module and remains re-exported through the neural module. Its IDs and serialized specification are unchanged: 256 byte IDs, BOS 256, EOS 257, PAD 258.

The reference byte-pair fitter begins with those byte IDs, counts adjacent pairs separately within each training document, merges the most frequent pair, and breaks frequency ties by lexical token-ID order. New IDs start at 259; special tokens never merge. Fitting uses the train partition only and
```

```text
{
  "uri": "aim:docs/PRETRAINING.md",
  "version": "sha256:0f46c424e763e55d05ab8dae57f1c24c5e323f3d221ea73c6fd0553e6546e052",
  "title": "Phase 3A — From-scratch corpus and pretraining infrastructure",
  "source_id": "lib-src-68b806a108ce1d20d297100c6af1804b8d196a44819a7f5e5376550b50401cca",
  "chunk_id": "lib-chunk-519774eb666568fa7716d806d7b03f7e94e3521c441083b574ef199fe00624c0",
  "start": 2688,
  "end": 3712,
  "evidence_id": "ev-4a4d92c80e12ad26142d7ab4"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 6 — UNVERIFIED

```text
ency ties by lexical token-ID order. New IDs start at 259; special tokens never merge. Fitting uses the train partition only and records corpus/train-content hashes. Its engineering limits are 65,536 training bytes, 128 merges and 256 bytes per merged token. This simple implementation is not a production-scale tokenizer trainer.

Encoding applies the saved merge sequence; decoding concatenates the saved byte strings. Byte fallback gives UTF-8 coverage without an unknown token. The comparison checks exact round trips and bytes per token, with per-document prose/code/math/Unicode records on validation only. Test text is not available to fitting or comparison. The fixture's repeated template families cross splits deliberately; it is an integration workload, not a family-generalization benchmark.

Compression alone does not select a tokenizer. A scientific comparison needs licensed representative data, downstream quality, equalized compute and byte-normalized likelihood measures. Raw per-token NLL cannot be compa
```

```text
{
  "uri": "aim:docs/PRETRAINING.md",
  "version": "sha256:0f46c424e763e55d05ab8dae57f1c24c5e323f3d221ea73c6fd0553e6546e052",
  "title": "Phase 3A — From-scratch corpus and pretraining infrastructure",
  "source_id": "lib-src-68b806a108ce1d20d297100c6af1804b8d196a44819a7f5e5376550b50401cca",
  "chunk_id": "lib-chunk-5e8b1d0e4ada30124f9a8965192ecd526f6ebc0a2410afca115c8d977e77ea15",
  "start": 3584,
  "end": 4608,
  "evidence_id": "ev-8447fbf71157bc4eb40b3235"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 7 — UNVERIFIED

````text
heckpoint-inspect runs/<job>/job/checkpoints
.venv/bin/python -m aim checkpoint-inspect runs/<job>/job/checkpoints/step-000006 --corpus runs/<intake>/corpus.json
.venv/bin/python -m aim distributed-pretrain --corpus runs/<intake>/corpus.json --ranks 2 --config configs/<longer-run>.json --resume runs/<job>/job/checkpoints/step-000006
.venv/bin/python -m aim distributed-export --checkpoint runs/<job>/job/checkpoints/step-000006
.venv/bin/python -m aim.distributed_reproduce --export reports/<new-directory>
```

The preflight checks training/validation object integrity, ownership, effective batch size and the analytical FP32 Adam parameter/gradient/moment floor, `16*N` bytes **per rank**. It excludes activations/runtime and is not an allocation guarantee. It reports local free disk without certifying cluster storage.

The initialization exporter creates an explicitly initialization-only checkpoint linked to its source bundle. The existing SFT path accepts it only with matching byte vocabulary and model dimensions
````

```text
{
  "uri": "aim:docs/DISTRIBUTED_PRETRAINING.md",
  "version": "sha256:cfeaf03b9735afd4b700450a92010e12acc3a3767728008ebce7d112ab0b5aa0",
  "title": "Phase 3B — Distributed corpus training and recovery",
  "source_id": "lib-src-dfb6a9189b47468389cfbc2552b365fbfb4c8877e33bc6e62f8999b8e8c3acaa",
  "chunk_id": "lib-chunk-bec7cb01ace12dd4c8e7d765ea80663713ea5149e0474ef27b9b37ab12ecd296",
  "start": 8064,
  "end": 9088,
  "evidence_id": "ev-bc549edcaff85a6ad99f2f4f"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 8 — UNVERIFIED

```text
mputes all features from the permitted input and rejects altered
values. Row IDs, families, source records and groups are attribution/splitting
metadata; only the recomputed numerical vectors enter tensor training. This is
an auditable code boundary, not an operating-system sandbox.

## Forecast and decision objectives

For the defined binary future-verifier event Y, p=sigmoid(z). Fit log loss
`-mean(Y*log(p)+(1-Y)*log(1-p))` or Brier `mean((p-Y)^2)` separately. Each
feature/loss/seed run uses a fresh random model, 600 sampled batches of 64, and
the same source training worlds. The five/seven input networks have 225/289
parameters. These numbers establish a mechanism comparison, not final sizing.

Temperature scaling substitutes sigmoid(z/T), fitting only calibration log loss
over the registered temperature grid. It cannot add information absent from z.
Raw and scaled forecasts are both reported; holdout never selects temperature,
steps or a preferred checkpoint. A constant-half and training-base-rate forecas
```

```text
{
  "uri": "aim:docs/JUDGE_SHIFT.md",
  "version": "sha256:ff0670e4897d16fc22a7c724a4098b2fa977961854943e1730bb2192e4fd3124",
  "title": "Separate Judge forecasts and decisions under shift",
  "source_id": "lib-src-d4065518beea83b70aeda75874a6f0444eff1bf840483244859af2c52e2f69af",
  "chunk_id": "lib-chunk-8dd070901be863897b92dcb4067fd00b7e7b61bd1994b7e5d71bf81688a084eb",
  "start": 2688,
  "end": 3712,
  "evidence_id": "ev-69d2b0b647e0a12ea7c486ce"
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

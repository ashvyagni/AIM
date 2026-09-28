# AIM local evidence review

## Question

```text
checkpoint resume exact tokenizer BPE
```

Source excerpts below are **UNVERIFIED assertions**. A PASS applies only to attribution.

## Excerpt 1 — UNVERIFIED

```text
checkpoint linked to its source bundle. The existing SFT path accepts it only with matching byte vocabulary and model dimensions; it is not a distributed-resume file. BPE→SFT remains an explicit future adapter.

For an authorized multi-host trial, invoke `aim.distributed_pretrain` under a separately configured `torchrun`, supplying `--config`, `--corpus`, `--job` and optional `--tokenizer`/`--resume` paths. Do not reuse the local launcher's loopback override for real hosts. No such deployment is performed by this phase.

## Acceptance gates, fixed before the full reproduction

All tests must pass in the pinned environment. Partition ownership must be disjoint/complete, and wrong-rank/world cursors rejected. Two-rank byte/BPE training must complete with equal replicas. Byte accumulation must agree with the serial global-token oracle within 2e-6. Split continuation and recovery from both publication-boundary faults must be exact. A four-local-rank smoke job must pass data/checkpoint audits. Initialization expor
```

```text
{
  "uri": "aim:docs/DISTRIBUTED_PRETRAINING.md",
  "version": "sha256:cfeaf03b9735afd4b700450a92010e12acc3a3767728008ebce7d112ab0b5aa0",
  "title": "Phase 3B — Distributed corpus training and recovery",
  "source_id": "lib-src-dfb6a9189b47468389cfbc2552b365fbfb4c8877e33bc6e62f8999b8e8c3acaa",
  "chunk_id": "lib-chunk-9e3f9afedc41aa2a31b2ecb284a7b6e3d7cd49fc5512d4585c9d11b32cad351a",
  "start": 8960,
  "end": 9984,
  "evidence_id": "ev-e55b731f95e79723465978e6"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 2 — UNVERIFIED

````text
s the parent hash. BPE→existing SFT is rejected until the supervised pipeline explicitly supports that tokenizer; no embeddings or output IDs are silently remapped. Pretraining checkpoints cannot masquerade as already trained research-format checkpoints.

## Commands

From `model-1/`, using the existing pinned environment:

```sh
.venv/bin/python -m aim corpus-fixture --output runs/my-fixture
.venv/bin/python -m aim corpus-intake --manifest runs/my-fixture/manifest.json
.venv/bin/python -m aim tokenizer-compare --corpus runs/<intake-run>/corpus.json
.venv/bin/python -m aim pretrain --corpus runs/<intake-run>/corpus.json
.venv/bin/python -m aim pretrain --corpus runs/<intake-run>/corpus.json --tokenizer runs/<comparison-run>/bpe-tokenizer.json
.venv/bin/python -m aim pretrain --corpus runs/<intake-run>/corpus.json --config configs/<longer-run>.json --resume runs/<pretrain-run>/checkpoint.pt
.venv/bin/python -m aim.pretrain_reproduce --export reports/<new-directory>
```

The reproduction generates only the vers
````

```text
{
  "uri": "aim:docs/PRETRAINING.md",
  "version": "sha256:0f46c424e763e55d05ab8dae57f1c24c5e323f3d221ea73c6fd0553e6546e052",
  "title": "Phase 3A — From-scratch corpus and pretraining infrastructure",
  "source_id": "lib-src-68b806a108ce1d20d297100c6af1804b8d196a44819a7f5e5376550b50401cca",
  "chunk_id": "lib-chunk-eda17f87450f1433efc2853e0e2f9b3137bb827e0bf0bf45011e3c99c1044f18",
  "start": 8064,
  "end": 9088,
  "evidence_id": "ev-05d311ea0a5cc037c7a4653b"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 3 — UNVERIFIED

```text
ly resume, and interruption recovery must match uninterrupted training on all recorded state checks. The trainer must not read test text. The byte-token SFT bridge must load correctly and the BPE bridge must fail explicitly.

Report actual timing and losses, including failed-run status; do not set a model-quality gate on this constructed corpus. Short local timings are not sustained throughput and cannot justify 70–84-node VIT scaling. No new benchmark suite replaces earlier canonical evaluations.

## Next build and remaining gates

Phase 3B implemented [rank-owned corpus training and checkpoint recovery](DISTRIBUTED_PRETRAINING.md); Phase 3C added operator observation and corpus review/release preparation. Next extend explicit tokenizer/vocabulary compatibility into supervised training, starting with a BPE→SFT bridge. Before actual 100M–300M proxy pretraining, obtain a reviewed representative corpus, scalable tokenizer/loader tooling, meaningful held-out evaluation and sustained hardware/network measurements
```

```text
{
  "uri": "aim:docs/PRETRAINING.md",
  "version": "sha256:0f46c424e763e55d05ab8dae57f1c24c5e323f3d221ea73c6fd0553e6546e052",
  "title": "Phase 3A — From-scratch corpus and pretraining infrastructure",
  "source_id": "lib-src-68b806a108ce1d20d297100c6af1804b8d196a44819a7f5e5376550b50401cca",
  "chunk_id": "lib-chunk-63f37f0d9b022676f048e5b47173672eab214957652be24bfb5e2cdc190a585a",
  "start": 9856,
  "end": 10880,
  "evidence_id": "ev-3b5916eeb044f9ea0ab19cec"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 4 — UNVERIFIED

```text
.

Resume restores the common state plus the caller's rank state. Configuration, source corpus, partition, tokenizer and world-size changes are rejected. Exact continuation is scoped to the recorded local CPU environment; changed devices, libraries, kernels or topology may alter numerical behavior. Mid-microbatch recovery is not supported: recovery starts at a completed optimizer-step checkpoint.

## Failure drills and audits

Two explicit test modes raise an exception on one worker after its rank file is saved:

- `before_publish`: the in-progress step must remain INCOMPLETE; recover from the previous published step.
- `after_publish`: the complete checkpoint remains loadable even though the job subsequently fails; recover from that published step.

The worker exception, peer/launcher failure and all checkpoint directories are retained. A terminated peer may lack a final per-rank status file; the supervisor's FAILED record and launcher log remain authoritative for job failure. These drills exercise worker-ex
```

```text
{
  "uri": "aim:docs/DISTRIBUTED_PRETRAINING.md",
  "version": "sha256:cfeaf03b9735afd4b700450a92010e12acc3a3767728008ebce7d112ab0b5aa0",
  "title": "Phase 3B — Distributed corpus training and recovery",
  "source_id": "lib-src-dfb6a9189b47468389cfbc2552b365fbfb4c8877e33bc6e62f8999b8e8c3acaa",
  "chunk_id": "lib-chunk-8f0367c23c1de3d4775eeb10d819cb58eebc9ffd4a47f57a85fa060be0a40483",
  "start": 6272,
  "end": 7296,
  "evidence_id": "ev-3bdf5ab303602564058434ee"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 5 — UNVERIFIED

```text
 three SFT budgets. Checkpoint continuation retains the original frozen reference. The runner writes prepared data, frozen probes, tokenizer specs, per-checkpoint diagnostics, training runs, actual final-budget Controller episodes, and positive reference controls. It creates no research test/OOD file. Existing tokenizer corpus partitions retain their historical role; BPE fitting reads its train split only.

`audit_study` rechecks all planned arm/budget/case identities, reconstructs initial weights, binds checkpoints and probes, decodes stored generation IDs, re-scores candidates, recomputes aggregates, compares final diagnostic generation with actual Controller traces, and replays state ledgers. It checks saved likelihood summaries for internal consistency; it does not independently recompute neural logits. Hashes detect ordinary changes but are not authenticated signatures.

Export is explicit and creates a new directory. It copies only this authored study's diagnostics/probes/tokenizers and summary, with so
```

```text
{
  "uri": "aim:docs/LANGUAGE_FITTING.md",
  "version": "sha256:8db39a232634a32c130df2cbab8379da9bd1d95a1033ec533d29b55ee1a9a4fb",
  "title": "Language fitting diagnostics",
  "source_id": "lib-src-344676b01d020c3e0f14ee58fb3e7f9b68bf4daa4cd48e97816c932e9ef1488e",
  "chunk_id": "lib-chunk-996b556b50c611e1f56062f60139269fd82885fd2562e970b1ad21e99887f158",
  "start": 3584,
  "end": 4608,
  "evidence_id": "ev-f543b894ccf30c31f198bb4c"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 6 — UNVERIFIED

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

## Excerpt 7 — UNVERIFIED

```text
actions, observable state, time-dependent outcomes, costs, delayed feedback and off-policy evaluation. Keep that proposed method in a new experiment/decision record.

## Stage transitions, checkpointing and failure handling

`--initialize` loads only compatible AIM-owned LM weights, creates a new optimizer and freezes a new reference. `--resume` instead checks stage/config/dataset identity, restores optimizer/reference/torch RNG and advances from the recorded step. The deterministic batch order follows step and seed. Every checkpoint has a content-hash sidecar. The user-provided path, parent hash and exported dataset accompany the run.

There is no current random Python/numpy sampling in the update loop; those libraries are seeded, and procedural generation is deterministic. A future stochastic dataloader must preserve its own RNG/cursor and worker state before claiming exact resume. Cross-device and cross-library-version bitwise continuation is not promised.

Nonfinite loss/gradient, incompatible checkpoint,
```

```text
{
  "uri": "aim:docs/TRAINING.md",
  "version": "sha256:2e3c8d3b769f93a79ad185bb48d63247be418aaa981fa69aee5c4e83546c7bcc",
  "title": "Model-1 training specification",
  "source_id": "lib-src-490dcc1789716918e289537e34b05ecebe53eab1cd208b34fa10a4ca757ed8cc",
  "chunk_id": "lib-chunk-1d6bccd44255d2eb891fd96137f0b42689d1f44ab8bfd96458c313be102ff04e",
  "start": 10752,
  "end": 11776,
  "evidence_id": "ev-e962fd40291da81e8d380173"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 8 — UNVERIFIED

```text
sed.
The fixed new-study trainer does not yet support checkpoint resume.

Read root `summary.json`, `tests.log`, `dataset-stage.json`, `training-stage.json`,
the training `frozen.json`, and evaluation `metrics.json`/`predictions.json`.
Configuration: `configs/judge-shift.json`. Results use the exact version and
dataset hashes; never overwrite a historical experiment or reuse its exposed
holdout for selection.

## Information boundary

`judge_shift_features.extract()` accepts exactly `observations`, `coefficients`
(the proposed candidate), and `target_x`. It rejects extra fields, nonfinite or
Boolean numbers, duplicate coordinates and observations at/after the target.
The runtime adapter projects these fields from state; it does not pass claims,
tool outcomes, world labels or hidden generator coefficients to the network.
Claim identity and candidate prediction must agree before a forecast is attached.

Five features preserve the prior Judge baseline: candidate coefficient count,
observation count, mean normali
```

```text
{
  "uri": "aim:docs/JUDGE_SHIFT.md",
  "version": "sha256:ff0670e4897d16fc22a7c724a4098b2fa977961854943e1730bb2192e4fd3124",
  "title": "Separate Judge forecasts and decisions under shift",
  "source_id": "lib-src-d4065518beea83b70aeda75874a6f0444eff1bf840483244859af2c52e2f69af",
  "chunk_id": "lib-chunk-d1dd526fa10e1df5b472f23b5fea193a1874913a4857ad55006c95fc6744cf2f",
  "start": 896,
  "end": 1920,
  "evidence_id": "ev-0059d5bdf381d18ed8822985"
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

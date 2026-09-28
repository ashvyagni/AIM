# AIM local evidence review

## Question

```text
VIT CPU cluster memory 7B DDP
```

Source excerpts below are **UNVERIFIED assertions**. A PASS applies only to attribution.

## Excerpt 1 — UNVERIFIED

```text
# Hardware and cluster engineering protocol

## Known versus reported

**Reported by project owner:** 70–84 lab desktops, each with 32 GB RAM, a 13th-generation Intel i9 and Intel UHD Graphics 770. Exact CPU SKU, core counts, usable RAM, OS, disk, network fabric, scheduling rights, simultaneous availability and sustained throughput have not been audited here. The delivered audit JSON is an unfilled template, not a measurement.

70×32 to 84×32 gives **2,240–2,688 nominal GB across the fleet**. That aggregate is not one address space. Standard DDP replicates the model and optimizer on each rank; it does not combine node memory to fit a larger model. Intel integrated graphics are not treated as validated training accelerators by this implementation. No supported precision or sustained GPU rate has been measured for them.

**Measured in this phase:** a macOS arm64 host reporting 8 logical CPUs, using FP32 CPU kernels and one torch thread per process. CPU marketing-name lookup was unavailable in the restricted ben
```

```text
{
  "uri": "aim:docs/HARDWARE_AND_CLUSTER.md",
  "version": "sha256:941ac209ea90192bd70128721971e7db1355825059917acee99f588509be7771",
  "title": "Hardware and cluster engineering protocol",
  "source_id": "lib-src-ed4b2b930c4c34e24e39b4ba4e4b6210b360f615523827299d03bbabcbc0e22c",
  "chunk_id": "lib-chunk-afc060ec8d16b1a3d3574caaa6510ecc63c0e276fea38cd8f586ff74196f7a12",
  "start": 0,
  "end": 1024,
  "evidence_id": "ev-53dbf3344b0a841ed45b9e4c"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 2 — UNVERIFIED

````text
l CPUs, using FP32 CPU kernels and one torch thread per process. CPU marketing-name lookup was unavailable in the restricted benchmark, and is recorded null. The host is not claimed to match any VIT desktop. Timings cannot be extrapolated to the lab.

## Implemented benchmark commands

From `model-1/`:

```sh
.venv/bin/python -m aim benchmark
.venv/bin/python -m aim.distributed_launch --ranks 1
.venv/bin/python -m aim.distributed_launch --ranks 2
```

The CPU benchmark includes two warmup iterations and five measured samples for a 90,624-parameter model, batch 2, sequence length 64. It measures a complete training update, no-gradient prefill, byte tokenization, 256×256 matrix multiplication, checkpoint write/load equality and peak RSS. The checkpoint write time is not a durable-disk/fsync or filesystem bandwidth guarantee.

The local DDP benchmark uses the same model, batch 2 per rank, sequence length 32, two warmups and five samples. It measures max-rank update time, verifies all-reduce values, and checks ex
````

```text
{
  "uri": "aim:docs/HARDWARE_AND_CLUSTER.md",
  "version": "sha256:941ac209ea90192bd70128721971e7db1355825059917acee99f588509be7771",
  "title": "Hardware and cluster engineering protocol",
  "source_id": "lib-src-ed4b2b930c4c34e24e39b4ba4e4b6210b360f615523827299d03bbabcbc0e22c",
  "chunk_id": "lib-chunk-bf3d351328faea85c2393ac583d7e4667680d95d8702f225c8519812b391e31f",
  "start": 896,
  "end": 1920,
  "evidence_id": "ev-294f0c5934b870210d0ce886"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 3 — UNVERIFIED

```text
# Phase 3B — Distributed corpus training and recovery

## Build scope

Extend the existing CPU/Gloo infrastructure to train the native decoder on the Phase 3A corpus stream. Keep the 2M-parameter allocation limit, fixed world size, separate post-training objectives and unchanged research-loop architecture. The local launcher permits 1, 2 or 4 processes on loopback. Direct worker entry points support a declared shared-filesystem deployment, but no physical VIT host or network is validated here.

This adds actual corpus-driven DDP updates, gradient accumulation, deterministic ownership, complete-rank checkpoints, recovery, data replay audits, a serial mathematical reference and operator commands. It does not implement model/optimizer sharding, automatic node discovery, elastic membership, remote deployment or a production scheduler.

## Data ownership

`aim-document-partition-v1` sorts training document IDs and gives rank r every W-th document starting at r. It records the full assignment, byte counts, corpus h
```

```text
{
  "uri": "aim:docs/DISTRIBUTED_PRETRAINING.md",
  "version": "sha256:cfeaf03b9735afd4b700450a92010e12acc3a3767728008ebce7d112ab0b5aa0",
  "title": "Phase 3B — Distributed corpus training and recovery",
  "source_id": "lib-src-dfb6a9189b47468389cfbc2552b365fbfb4c8877e33bc6e62f8999b8e8c3acaa",
  "chunk_id": "lib-chunk-fc79b304e98f18c6edb7d7d14e65ed8c2ec673d8ce14e2b2bbbdcf1b39855642",
  "start": 0,
  "end": 1024,
  "evidence_id": "ev-11ddf0acfda3568eac680370"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 4 — UNVERIFIED

```text
ary remapping. Any model/optimizer sharding, larger allocation, accelerator precision, elastic world-size change or new mixture policy requires a separately specified experiment. The current CPU DDP implementation is a reproducible systems foundation, not the final cluster strategy.

```

```text
{
  "uri": "aim:docs/DISTRIBUTED_PRETRAINING.md",
  "version": "sha256:cfeaf03b9735afd4b700450a92010e12acc3a3767728008ebce7d112ab0b5aa0",
  "title": "Phase 3B — Distributed corpus training and recovery",
  "source_id": "lib-src-dfb6a9189b47468389cfbc2552b365fbfb4c8877e33bc6e62f8999b8e8c3acaa",
  "chunk_id": "lib-chunk-fb6780e636132e3a13ea45b888d813fc8d6053771b0d30b1ab1fcf6efd509505",
  "start": 10752,
  "end": 11036,
  "evidence_id": "ev-4f05af846e27b1cfa93ea231"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 5 — UNVERIFIED

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
acked runs. It does not modify the checkpoint or research memory.

## Measurements and boundaries

Each probe records gold-prefix response NLL, response-token accuracy, gold-position EOS probability, reference/argmax token IDs, response-length coverage, generated IDs/text, termination reason and first reference divergence. EOS counts as a response target. The aggregate NLL weights tokens; the EOS diagnostic is a mean over examples. Direct raw token-NLL comparisons across vocabularies are not a quality ranking.

JSON syntax, output-contract validity and semantic checks are separate. Parser positions refer to character/UTF-8 byte offsets where the parser supplies one. A missing closing delimiter can point to end-of-input; that is not a particular invalid generated token. Duplicate/schema errors may have null positions. Invalid generated BOS/PAD has a separate token index. A different valid polynomial expression can pass the checker without matching reference tokens.

The numerical diagnostic uses exact rational
```

```text
{
  "uri": "aim:docs/LANGUAGE_FITTING.md",
  "version": "sha256:8db39a232634a32c130df2cbab8379da9bd1d95a1033ec533d29b55ee1a9a4fb",
  "title": "Language fitting diagnostics",
  "source_id": "lib-src-344676b01d020c3e0f14ee58fb3e7f9b68bf4daa4cd48e97816c932e9ef1488e",
  "chunk_id": "lib-chunk-664e95b9b6eb67452d40680a606295baafabc16710d622ba95f70f550809962a",
  "start": 1792,
  "end": 2816,
  "evidence_id": "ev-f020a4062377f42d901ac196"
}
```

Attribution check: PASS. Character offsets refer to the unchanged original.

## Excerpt 8 — UNVERIFIED

```text
ndependent annotators, disagreement tracking, adjudication and data-use permission. Do not mark the included procedural preference experiment as a human RLHF result. No preference reward model or PPO trainer is implemented in this phase.

External RLVR intake deliberately fails until a dataset-specific verifier adapter is implemented and reviewed. Unknown verifier outcomes must remain missing labels, not automatic failures or successes.

## Objective 1: response-masked SFT

For prompt x and target response y, including EOS:

\[
L_{SFT}=-\frac{\sum_{i,t\in response_i}\log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})}{\sum_i |response_i|}.
\]

Prompt and right-padding positions have zero loss. A BOS token anchors the first prompt. Context overflow fails explicitly; there is no silent truncation. Tests check response token counts, padding independence and causal prefix invariance.

Delivered config: 60 steps, batch 8, AdamW learning rate 0.002, weight decay 0.01, seed 17, global gradient norm clipped at 1.0. FP32 CPU exe
```

```text
{
  "uri": "aim:docs/TRAINING.md",
  "version": "sha256:2e3c8d3b769f93a79ad185bb48d63247be418aaa981fa69aee5c4e83546c7bcc",
  "title": "Model-1 training specification",
  "source_id": "lib-src-490dcc1789716918e289537e34b05ecebe53eab1cd208b34fa10a4ca757ed8cc",
  "chunk_id": "lib-chunk-9e334c19aa248c5aba8f7f8e3af759435258c30c052321fc122d0acc6c691c9a",
  "start": 5376,
  "end": 6400,
  "evidence_id": "ev-253e62b9abb2feb0c039e3b6"
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

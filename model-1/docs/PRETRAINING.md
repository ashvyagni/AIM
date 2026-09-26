# Phase 3A — From-scratch corpus and pretraining infrastructure

## What this build decides

Implement local corpus intake, reversible tokenizer candidates, document-at-a-time next-token training and exact CPU continuation before a larger model run. Keep the 2M-parameter engineering allocation guard. Byte tokenization remains the compatibility baseline; the new byte-pair tokenizer is an explicitly bounded experiment. No final tokenizer, corpus mixture or parameter count is selected here.

This supports the approved native Researcher architecture. It does not replace the Controller, Judge, verifier or memory components, and does not combine their training objectives.

## Corpus intake contract

`aim-corpus-manifest-v1` contains a version and a document list. Each document requires:

- Unique ID, relative local path, SHA-256, declared split and group.
- Source URI and version, plus a content category: prose, code, math or Unicode.
- Rights fields: license string, explicit Boolean `training_allowed: true`, basis and reviewer.
- Privacy-review fields: approved status, basis and reviewer.

These are recorded declarations, **not independent legal or privacy certification**. Intake never invents permissions from availability or a filename. External corpus owners/reviewers must supply accurate declarations. The built-in fixture uses project-generated text with a construction-based declaration; it contains no real-person records or imported passages. No external corpus is downloaded or approved by this phase.

Paths must stay within the manifest directory. Absolute paths, parent traversal, symlinks, non-files, missing/oversized content, mismatched hashes, non-UTF-8, blank/NUL-containing documents, duplicate IDs and cross-split declared groups are rejected. Exact content and NFC-plus-collapsed-whitespace duplicates are rejected across the entire intake; nothing is silently dropped or reassigned. This is not semantic/near-duplicate detection. Incorrect group declarations can still hide family leakage.

The reference intake caps each document at 262,144 bytes, total text at 16 MiB, and count at 10,000. It copies original bytes into immutable named objects and emits a fingerprinted index only after successful validation. Normalization is used for duplicate detection, never to rewrite training text. Source versions, original paths, rights and review declarations remain in the local index. A failed intake keeps its run status, accepted-object history and traceback; no completed index is published.

Loaders verify index identity and recheck each object's hash and metadata when loading it. Hashes are corruption checks, not signatures against a privileged actor who rewrites both content and metadata. Do not publish external/private corpus objects or manifests through the engineering fixture exporter.

## Tokenizer candidates

The existing UTF-8 byte tokenizer now lives in a dependency-free module and remains re-exported through the neural module. Its IDs and serialized specification are unchanged: 256 byte IDs, BOS 256, EOS 257, PAD 258.

The reference byte-pair fitter begins with those byte IDs, counts adjacent pairs separately within each training document, merges the most frequent pair, and breaks frequency ties by lexical token-ID order. New IDs start at 259; special tokens never merge. Fitting uses the train partition only and records corpus/train-content hashes. Its engineering limits are 65,536 training bytes, 128 merges and 256 bytes per merged token. This simple implementation is not a production-scale tokenizer trainer.

Encoding applies the saved merge sequence; decoding concatenates the saved byte strings. Byte fallback gives UTF-8 coverage without an unknown token. The comparison checks exact round trips and bytes per token, with per-document prose/code/math/Unicode records on validation only. Test text is not available to fitting or comparison. The fixture's repeated template families cross splits deliberately; it is an integration workload, not a family-generalization benchmark.

Compression alone does not select a tokenizer. A scientific comparison needs licensed representative data, downstream quality, equalized compute and byte-normalized likelihood measures. Raw per-token NLL cannot be compared between vocabularies. The BPE pretraining path verifies its fitting corpus and train hashes match the chosen corpus; vocabulary conversion is never implicit.

## Streaming and boundary semantics

`TokenStream` loads/encodes one bounded document at a time and retains metadata plus one document token cache. The BPE fitting stage separately holds its small bounded training set. This is not a distributed, shuffled or zero-copy streaming system.

Documents are ordered by ID and encoded as `[BOS, bytes-or-merged-tokens, EOS]`. Training repeats the train partition. Validation is finite and reads a configured prefix. Each packed window predicts the next token at each position. A one-token overlap between consecutive windows prevents dropping transitions. Partial finite windows use PAD; BOS and PAD targets are excluded from the loss. EOS targets are trained.

**Causal attention may cross document boundaries inside a packed window.** BOS supplies a boundary signal, and EOS→BOS prediction is masked, but this is not block-diagonal attention or independent-document attention. Position indices reset per context window. A future isolation/shuffling/mixture policy must be versioned and evaluated explicitly.

The cursor records corpus fingerprint, tokenizer hash, split, repeat mode, next document, next token offset, epoch and carried token with its document/offset/epoch provenance. Restoration validates that the carry is exactly the preceding token implied by the cursor. Batch hashes include token arrays, loss masks and token provenance. Fixed ordering plus this cursor specifies continuation without a random data-loader state.

## Objective and checkpoints

The native random-init causal decoder uses CPU FP32 and AdamW. With target mask m, the objective is:

`L = - sum_{b,t} m[b,t] log p_theta(y[b,t] | x[b,0:t]) / sum_{b,t} m[b,t]`.

This is ordinary next-token pretraining; it supplies no RLHF/RLVR/Judge reward. Nonfinite loss or gradients fail the run. Training is bounded to 1,000 steps, batch at most 16, context at most 512 and 2M parameters. The smoke config is substantially smaller. These are harness limits, not scientific scaling recommendations.

Periodic and final checkpoints record weights, optimizer, Torch RNG, global step, scored-token count, complete stream cursor, corpus identity, tokenizer specification, stable configuration and parent checksum. Resume may increase the total step target but rejects other configuration, data or tokenizer changes. Existing checkpoints are never overwritten. Publication uses an exclusive weight file followed by its hash sidecar; a partial file or missing sidecar cannot load. Prior completed checkpoints remain usable. This is not an fsync/power-loss durability guarantee or a transaction across multiple nodes.

The reproduction includes an intentional application exception immediately after the first periodic checkpoint is fully saved. That run must have status FAILED and retain its traceback. Recovery must match the uninterrupted run's tensors, optimizer, RNG, cursor, token count, continued batch hashes and validation metrics. This tests a defined application interruption, not SIGKILL, OS crash, device loss or distributed recovery.

`load_pretrained` restores either supported tokenizer. The existing SFT trainer can initialize from a byte-token pretraining checkpoint with exactly matching dimensions and vocabulary. It creates a fresh optimizer/reference and preserves the parent hash. BPE→existing SFT is rejected until the supervised pipeline explicitly supports that tokenizer; no embeddings or output IDs are silently remapped. Pretraining checkpoints cannot masquerade as already trained research-format checkpoints.

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

The reproduction generates only the versioned 20-document engineering fixture (12 train, four validation, four test), compares byte and 32-merge BPE, performs full and split/resumed runs, conducts the retained interruption drill, and checks the byte-pretraining→SFT transition. Export directories are exclusive; logs, failures, intermediate runs and weights remain on disk.

## Acceptance checks fixed before the complete build run

All legacy/new tests must pass with neural dependencies installed. Corpus intake must reject unpermitted declarations, corruption and declared split leakage. Tokenizers must round-trip all fixture and adversarial Unicode cases. Finite token packing must preserve every eligible transition, including boundaries and the final partial window. Both tokenizer variants must exactly resume, and interruption recovery must match uninterrupted training on all recorded state checks. The trainer must not read test text. The byte-token SFT bridge must load correctly and the BPE bridge must fail explicitly.

Report actual timing and losses, including failed-run status; do not set a model-quality gate on this constructed corpus. Short local timings are not sustained throughput and cannot justify 70–84-node VIT scaling. No new benchmark suite replaces earlier canonical evaluations.

## Next build and remaining gates

Next add deterministic multi-shard partitioning and data-parallel cursor/checkpoint recovery at the same small local allocation, with a declared process/rank/world-size contract. Continue physical-node audit preparation. Before actual 100M–300M proxy pretraining, obtain a reviewed representative corpus, scalable tokenizer/loader tooling, meaningful held-out evaluation and sustained hardware/network measurements. MoE, larger dense models, precision choices and fleet feasibility remain unresolved.

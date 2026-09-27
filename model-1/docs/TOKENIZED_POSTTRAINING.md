# Phase 3D — Tokenizer-compatible post-training

**Status: locally audited, 2026-09-27.** The [executed audit](../reports/phase-3d-audit.md) passed 208 tests, all six byte/BPE stage-continuation comparisons, two application-interruption recoveries and a two-worker BPE initialization bridge. Twelve actual Controller episodes replayed; the eight neural cases produced no valid hypotheses. Four adversarial findings were fixed with retained failure logs. The initial [build-only record](../reports/phase-3d-build.md) preserves the earlier requested deferral. The separate legacy byte path remains available; no learned default or larger scale is approved.

## Components

| Module | Responsibility |
|---|---|
| `language_contract.py` | Exact tokenizer identity, task/output schemas, compatible initialization/resume and versioned checkpoint loading |
| `response_encoding.py` | Separate prompt/response encoding, response-only masks, padded batches and encoding-accounting manifests |
| `language_data.py` | Train/validation-only procedural, external and structured inputs; bounded verifier reward adapters |
| `language_training.py` | Independent SFT, preference/DPO and bounded RLVR stages, periodic checkpoints, frozen references and batched validation |
| `generation.py` | Greedy generation records with token IDs, termination reason and strict special-token/UTF-8 behavior |
| `language_artifacts.py` | Read-only checkpoint descriptions, generation runs and non-executing stage compatibility plans |
| `language_reproduce.py` | Full regression, byte/BPE stage-chain and continuation audit runner |
| `language_integration.py` | Actual Controller episodes, read-only replay, retained interruption drills and distributed BPE initialization bridge |

The only legacy integration hooks are explicit `tokenized_lm` dispatch in `load_lm` and an output-contract check in the numerical Researcher adapter. Versioned symbolic checkpoints use the existing symbolic adapter contract. No Controller/Judge/verifier truth rule changes. The Judge's proper-score/calibration trainers stay separate from language-model objectives.

## Tokenizer and boundary identity

The tokenizer is either supplied explicitly, inherited from an initialized/resumed checkpoint, or the byte baseline when starting without either. Explicit and inherited specifications must match completely: equal vocabulary size alone is insufficient. The full BPE merge list and fitting provenance are part of the identity. Model vocabulary is derived from that tokenizer, and an explicit conflicting model vocabulary is rejected. No embedding resizing or ID remapping is implemented.

The response convention is `aim-response-segments-v1`:

1. Prefix: BOS followed by `encode(prompt)`.
2. Response: independently `encode(response)` followed by EOS.
3. Predict next tokens and score only response tokens, including EOS.
4. Right padding has zero loss. Context overflow rejects the example; there is no truncation.

BPE can merge across a prompt/response boundary when encoding a concatenated string. This implementation deliberately prevents that by encoding the segments separately. For a tokenizer merging `a+b`, prompt `a` and response `b` remain distinct token segments. Training and inference prefix construction share this convention. It is versioned behavior, not a claim that this is the best segmentation policy.

The encoding manifest records each relevant example/variant's tokenized hash, input/response lengths and response bytes. It covers SFT responses, preference pairs and RLVR candidate strings; validation preference diagnostics are included when available. Resume binds its full hash. Raw token NLL across vocabularies remains unsuitable as a standalone quality comparison.

## Supported data/task contracts

- **Arithmetic:** existing authored train/validation fixtures. SFT, procedural preference and the existing three-candidate arithmetic verifier environment.
- **Symbolic:** existing versioned binomial fixtures and independent bounded checker reward. SFT, preference and finite-candidate RLVR; output is the existing symbolic Researcher schema.
- **Structured numerical research:** `aim-research-sft-v1`, the existing plain `aim-structured-research-v1` output contract, and SFT only. Worked/curriculum formats are not silently mapped into it.
- **External text:** explicit `aim-language-data-v1` train/validation files. SFT and preference only; external RLVR requires a separately reviewed verifier adapter.

External file fields are exactly `schema`, `version`, `train`, `validation`. Each row needs `id`, `group`, `prompt`, `response`, `rights` and `label_origin`. Preference stages additionally need distinct `chosen` and `rejected`. Optional row `split` must agree with its enclosing split. Optional `annotation_batch` is required for human labels. All other row fields reject, including undeclared holdout metadata. IDs and train/validation groups must be disjoint; exact prompt leakage is rejected. Human-label provenance remains a declaration requiring annotation review.

This path does not fit a tokenizer on supervised validation text. It consumes the separately fitted specification. Provenance identifies the fitting corpus; this alone does not prove the fitting corpus is uncontaminated or licensed. Use the corpus review/release tooling and separate supervised-data review. Test/OOD fields are rejected from the training input container. Whole-corpus semantic leakage detection is not provided by exact prompt/group checks.

## Independent objectives

SFT uses the response-token mean negative log likelihood, including EOS. Preference training reuses the existing DPO objective with a frozen stage-input reference and summed response log probabilities. RLVR reuses finite-candidate REINFORCE with a detached exact candidate baseline and candidate-distribution KL. These are the [existing separately specified objectives](TRAINING.md), now receiving versioned tokenization and encoding records.

There is no blended RLHF+RLVR+RLCD reward. Included preference labels remain programmatic. This does not add PPO, a preference reward model, open-ended RL, process supervision or sequential Judge-policy learning.

Symbolic candidate rewards reuse the existing task definition: checker PASS earns one, other/invalid candidates earn zero utility. This is the bounded action reward convention; an unknown verifier result is not added as a false factuality/calibration label. Candidate lists are privileged finite environments and cannot substantiate free-generation capability.

Scored-token counters mean response targets for SFT, chosen plus rejected response targets for DPO, and all candidate response targets for RLVR. Frozen-reference scoring is excluded from those counters. Thus they are not comparable compute budgets across stages. Verifier calls, sampled actions/rewards and KL remain separate metrics.

## Checkpoint and stage transitions

New checkpoints use kind `tokenized_lm`, schema `aim-tokenized-lm-v1`. They store model, optimizer, frozen reference, Torch RNG, step, token accounting, stable configuration, dataset/encoding hashes, complete tokenizer, output/research contracts and parent checkpoint hash. Hash sidecars and AIM random-init declarations use the existing storage primitives; they are not signatures.

Initialization loads compatible weights, creates a fresh optimizer and freezes a fresh reference. Supported sources are own single-process pretraining, explicit distributed initialization exports, same-task versioned language checkpoints, and compatible legacy byte checkpoints. Pretraining sources may initialize SFT only. Unsupported distributed bundles require the explicit exporter. Ambiguous legacy task/output identities reject import.

Resume accepts only a versioned checkpoint with the same task/stage, model, full tokenizer, stable config, dataset and encoding manifest. The target step must increase. It restores optimizer/reference/RNG/accounting. Periodic checkpoint files are uniquely named by completed step and retained; the final checkpoint has a separate filename. Mid-update recovery, optimizer sharding and distributed post-training are not implemented.

Record validation also binds stable configuration to the saved architecture, rejects nonfinite model/reference/optimizer state, checks reference tensor structure and checks CPU RNG shape/type. Adapter traces are initialized before decoding, so an exception cannot reuse a previous episode's trace. These checks arose from reproduced audit failures rather than assumed coverage.

The allocation guard remains two million parameters, context at most 512, batch at most 16 and at most 1,000 optimizer steps. The frozen reference adds memory. Larger candidate scales still require corpus, physical hardware and budget evidence.

## Runtime and generation behavior

`language-generate` records prompt hash, tokenizer/checkpoint hashes, token IDs, token budget, termination reason and text. It uses greedy decoding and preserves/restores the model's mode. EOS ends generation. Emitted BOS/PAD inside the response is an error rather than silently discarded content. Incomplete/invalid UTF-8 is recorded as an error instead of replaced text. A budget-limited valid string is recorded as `token_budget`, not a proof of complete reasoning.

The numerical Researcher adapter accepts a versioned LM only when it declares the structured numerical output contract. Free-text/arithmetic versioned models cannot be implicitly treated as trained numerical Researchers. The symbolic adapter requires its own existing contract. Malformed output still fails without a deterministic fallback. Neither a generation record nor a trained checkpoint can mark a claim VERIFIED.

`language-inspect` validates integrity/schema and describes lineage; it does not run tensor inference or certify training correctness. `language-stage-plan` lists compatible separate stages and required evaluation work; it cannot launch or promote a model.

## Operator commands

The audit exercised the training/runtime APIs and the complete reproduction command below. This list documents CLI entry points; it is not a claim that every CLI invocation was exercised individually. Use compatible model dimensions in the selected config; the supplied defaults do not resize an old pretraining model's context or weights.

```sh
.venv/bin/python -m aim language-train --config configs/language-sft.json --initialize <compatible-pretraining-checkpoint>
.venv/bin/python -m aim language-train --config configs/language-preference.json --initialize <versioned-SFT-checkpoint>
.venv/bin/python -m aim language-train --config configs/language-rlvr.json --initialize <versioned-preference-checkpoint>
.venv/bin/python -m aim language-train --config <same-stage-higher-total-steps>.json --resume <versioned-checkpoint>
.venv/bin/python -m aim language-inspect --checkpoint <versioned-checkpoint>
.venv/bin/python -m aim language-stage-plan --checkpoint <versioned-checkpoint>
.venv/bin/python -m aim language-generate --checkpoint <versioned-checkpoint> --prompt-file <prompt.txt> --max-new-tokens 64
```

An explicit `--tokenizer <spec.json>` is optional when initializing/resuming: the exact source specification is otherwise inherited. Starting random-init BPE SFT requires that explicit tokenizer. No pretrained tokenizer or weights are downloaded.

## Repeatable audit and acceptance scope

To reproduce in a new directory, run the targeted suites and full reproduction:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_language*.py' -v
.venv/bin/python -m aim.language_reproduce --export reports/<new-directory>
```

The second command will run all regressions, construct the existing engineering corpus, pretrain byte/BPE models locally, and run uninterrupted/partial/resumed SFT, DPO and RLVR chains. It must compare model, optimizer, frozen reference, RNG, steps/counters, data/encoding/tokenizer/config/task/stage state exactly before exporting a successful report. Any failure must remain retained and be reported before promotion.

The full runner also trains structured and symbolic Researcher adapters and executes their actual Controllers alongside deterministic positive controls. It retains two deliberately interrupted jobs, resumes their completed checkpoints, and audits a two-worker distributed-pretraining initialization export into BPE SFT. Export rechecks states, generation traces, checkpoint hashes and replay results. Preserve every failed run. These checks do not establish mid-update durability, generalization, independent proof of optimizer math or physical-cluster readiness.

The [current evidence](../reports/phase-3d-audit-evidence/manifest.json) identifies the exact audited revision. Eight neural cases exhausted the generation budget with invalid JSON. Next implement fitting/free-generation diagnostics on explicitly exposed training/validation probes, then register a fresh capability study. No sustained performance benchmark or model promotion follows from the present audit.

# Phase 3D — Executed language-path audit

Date: 2026-09-27. **Local systems audit passed after four fixes. Learned research capability remains unproven.** This report supersedes the pending validation status in the historical [build report](phase-3d-build.md); that report correctly records what had been executed at its own checkpoint.

## Evidence and source identity

- Baseline before this audit: `592dadc5c17369e8ad0dcbc4b704047529d85f78`.
- Audited source: `2ef34945f657f35af0d4b770104936ad8581e98d`.
- Source hash: `b00ec44062651c90cad8fb4e74fb85c7ee5150a63c8b2a2b93e3dcd9b9a66a8f`.
- Run: `model-1/runs/20260927T160636-phase-3d-audit-3892fe0c`.
- [Export manifest](phase-3d-audit-evidence/manifest.json), [results and generation traces](phase-3d-audit-evidence/results.json), [regression log](phase-3d-audit-evidence/tests.log), [retained finding logs](phase-3d-audit-findings/manifest.json).
- Full run: **59.249 seconds**; regression suite: **208 tests, zero skips, 48.934 seconds**. These are execution timings for miniature integration checks, not sustained throughput benchmarks.
- Environment: Python 3.12.14, PyTorch 2.8.0, NumPy 2.2.6, macOS 26.3.1 arm64, eight logical CPUs; one Torch compute thread per training worker. The bridge used two loopback Gloo workers on this machine. No VIT host was accessed.

## Built and repaired

The new integration runner connects actual tokenized Researchers to the numerical and symbolic Controllers, replays their evidence/state ledgers, exercises checkpoint-boundary interruption recovery, and carries a distributed BPE initialization artifact into supervised training. The full reproduction runs legacy regressions alongside the new language path.

Four adversarial checks failed before correction:

| Finding | Correction |
|---|---|
| External rows admitted undeclared metadata such as `holdout_answers` | Explicit row-field allowlist; extra metadata rejects |
| Symbolic decoding exceptions could leave a previous generation trace | Initialize each trace before generation; retain the current failure and generation record |
| Saved stable model configuration could disagree with saved architecture | Validate stable training configuration and bind the derived model configuration to the checkpoint |
| A nonfinite frozen reference tensor passed record validation | Check finite model/reference/optimizer state, reference tensor keys/shapes/dtypes and CPU RNG structure |

The initial existing suite passed 12 tests. The added adversarial suite then failed all four checks in retained run `20260927T160043-phase-3d-adversarial-audit-7275fe91`. After correction, the combined targeted suite passed 16 tests. These were real audit findings, separate from the intentional interruption drills below. The finding manifest records each source hash because the first targeted runs included uncommitted changes over the baseline commit.

## What passed

### Separate training stages and continuation

Both byte and train-fitted BPE tokenizers completed native pretraining followed by separate SFT, DPO preference and finite-candidate RLVR stages. Each stage ran four uninterrupted updates and a two-update run resumed to four. All six comparisons matched exactly on model, optimizer, frozen reference, Torch RNG, step, scored-token count, stable configuration, dataset hash, encoding manifest hash, tokenizer, encoding, task, stage and output contract. Parent lineage legitimately differs between full and resumed runs and is excluded from equality.

The byte model has **26,880 parameters** and vocabulary 259; BPE has **27,904 parameters** and vocabulary 291. Both use width 32, two layers, four query heads, two KV heads and FFN width 64. The arithmetic chain uses context 128. These sizes exercise interfaces; they select no production scale.

Validation below scores the same 32 fixed arithmetic responses in each stage. Preference labels are procedural. NLL is response-token NLL; do not use raw values to rank different tokenizers. Four-update outcomes are diagnostics, not a statistical capability study.

| Tokenizer / stage | Initial → final NLL | Final preference ordering |
|---|---:|---:|
| Byte SFT | 5.471775 → 5.315866 | 26/32 |
| Byte preference | 5.315866 → 5.350426 | 25/32 |
| Byte RLVR | 5.350426 → 5.373132 | 23/32 |
| BPE SFT | 5.699185 → 5.535898 | 17/32 |
| BPE preference | 5.535898 → 5.568758 | 19/32 |
| BPE RLVR | 5.568758 → 5.594249 | 19/32 |

SFT lowered the measured NLL; later stages increased it in these runs. The separate objectives need not improve this metric together. Finite-candidate reward and preference ordering do not establish unrestricted generation quality.

### Actual research loops

Four adapter checkpoints were trained for two updates: numerical and symbolic, each with byte and BPE tokenization. Numerical training consumed eight training/four validation examples; test cases entered only the Controller evaluator. The cases are already exposed engineering fixtures, not a fresh capability holdout. Numerical context was 512, symbolic context 256, and greedy generation was capped at 96 new tokens.

| Backend | Numerical episodes with verified claim | Symbolic episodes with verified claim |
|---|---:|---:|
| Deterministic reference | 2/2 | 2/2 |
| Byte neural | 0/2 | 0/2 |
| BPE neural | 0/2 | 0/2 |

All eight neural episodes exhausted the token budget and emitted malformed JSON; each produced zero hypotheses and zero verified claims. Their errors, raw outputs and token traces remain in the results. No fallback silently replaced them. The reference episodes each yielded one verified claim.

All 12 states replayed successfully. The numerical replay reused the Controller's final invariant checker to validate evidence and claim/check bindings; it is not a second independent implementation of numerical verification. The symbolic audit used the existing exact checker/ledger protocol. Passing replay confirms recorded behavior and scoped invariants, not scientific usefulness.

### Recovery and distributed initialization

Two BPE drills injected an application exception immediately after the completed step-two checkpoint: one SFT, one stochastic RLVR. Both runs remain **FAILED**, with their original logs/checkpoints. Resuming to step four matched the respective uninterrupted stage on all 14 fields. This establishes recovery at a completed application checkpoint in this CPU environment; it does not test power loss, partial disk writes or physical-node failure.

The two-worker BPE bridge trained two updates with **2,034 global scored targets**, audited disjoint ownership of 12 training documents, replayed four rank updates and confirmed equal replicas and valid cursor bindings. Its explicit initialization export entered two SFT updates with matching tokenizer and parent artifact hash. The bridge audit checks batch/cursor/count/certificate replay; the Phase 3B independent serial optimizer reference remains separate historical evidence.

## Decisions, limitations and remaining work

- Keep the opt-in versioned language path and separated objectives. No learned default, final tokenizer, model size or architecture is promoted by this audit.
- Keep the deterministic Controller, separate Judge, independent verifiers and provenance memory. No blended RLHF/RLVR/RLCD objective was introduced. Human preference collection and integrated research-loop policy learning remain unfinished.
- Preserve the two-million-parameter allocation guard. The reported 70–84 VIT desktops still require operator measurements, representative corpus approval and bounded physical-host trials before scaling decisions.
- Exact resume is demonstrated in one recorded software/CPU environment. Cross-platform numerical identity, elastic world size, optimizer sharding, sustained throughput and production recovery remain untested.
- Corpus fixtures, few-step training and exposed Controller cases establish integration coverage only. The complete scientific research system is not finished. Long-context literature retrieval, broader tools, useful learned proposals and calibration transfer remain open work.
- Canonical numerical and symbolic evaluation files and their hash sidecars were unchanged relative to the audit baseline.

## Recommended next experiment

Build a bounded **generation-failure diagnostic** for the versioned language path before another scale increase: freeze training/validation fitting probes; record response-length coverage, first invalid token, EOS behavior, syntax validity and verifier success; compare untrained and SFT checkpoints across fixed update budgets and multiple seeds. Separate teacher-forced response likelihood from free-generation outcomes. First establish that tiny examples can be fitted and emitted, then register fresh world-family test/OOD partitions before measuring generalization. Treat exposed cases here only as regression inputs. Preserve existing canonical suites.

A future constrained decoder can be a separately recorded inference experiment; it must still pass independent semantic verification and must not be credited as learned reasoning. Representative licensed data and real lab observations remain parallel prerequisites supplied through their existing review/operator protocols.

## Reproduction and delivery

From `model-1/`, in the recorded environment:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_language*.py' -v
.venv/bin/python -m aim.language_reproduce --export reports/<new-directory>
```

The complete runner requires local loopback worker sockets. Use a new export directory; do not overwrite this evidence. Run artifacts retain configs, dataset/tokenizer identities, source snapshots, checkpoints, metrics and failures. Git contains selected portable evidence, not the entire checkpoint tree. Portable log path redactions are declared with original and exported hashes.

Implementation commits: `cfa6d49` (contracts/traces and adversarial tests), `2ef3494` (Controller/recovery/bridge audit integration). The [exact changed-file inventory](phase-3d-audit-files.json) covers this audit and documentation against `592dadc`, excluding the inventory itself. The earlier build/status/inventory remain historical artifacts.

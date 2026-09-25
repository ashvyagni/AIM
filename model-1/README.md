# AIM Model-1

**Phase 1 foundation, structured Researcher training, and Phase 2A.1 arithmetic-supervision comparison.**

The default research loop uses a clearly identified deterministic polynomial Researcher and a separate rule-based or trained Judge. A native, randomly initialized transformer and an interchangeable neural Researcher adapter are implemented. The phase-1 arithmetic checkpoint failed structured generation. Phase 2A trained three research-specific checkpoints that emit mostly valid hypotheses but achieve only 3.13–10.94% verified success on the familiar holdout; their capability gate failed and they remain opt-in experimental backends.

Phase 2A.1 adds tested research-trainer continuation and a six-model comparison of plain versus worked arithmetic supervision. Mean test success was 5.21% versus 12.50%, but one seed regressed and both arms failed their numerical gate. Eight worked-arm familiar-case successes contained incorrect process steps, reinforcing the distinction between a checked target prediction and a checked reasoning trace.

See the [latest report](reports/phase-2a1-implementation.md), [comparison reproduction guide](docs/FINITE_DIFFERENCE_COMPARISON.md), and [real-checkpoint resume audit](reports/research-resume-audit.md). The complete suite has **77 passing tests, zero skips**. Historical reports preserve their original results and test counts.

## What runs

1. A question about a synthetic instrument is planned.
2. Local observation documents are retrieved into a versioned provenance store.
3. The Researcher proposes competing linear/quadratic hypotheses.
4. Bounded tools calculate predictions; the Judge forecasts success before measurement.
5. An independent exact numerical verifier checks a new synthetic measurement.
6. The Controller revises claims to VERIFIED, CONTRADICTED or UNKNOWN.
7. A final Markdown response links claims, checks and immutable source artifacts.

`VERIFIED` means the explicitly stated check passed for that claim. It never means that the global scientific hypothesis is proved.

## Setup

Run commands from `AIM/model-1/`. Python 3.12 is the recorded environment; the package declares Python ≥3.11. The standard-library research loop needs no third-party packages:

```sh
python3 -m aim loop
python3 -m aim evaluate
```

For all training and neural tests, create a local environment and install the recorded dependencies:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
```

If using `uv`, `uv venv .venv --python 3.12` and `uv pip install --python .venv/bin/python -r requirements-lock.txt` are equivalent setup routes. The delivered `.venv` was created with `uv`; it need not contain pip. Use the existing `.venv/bin/python` directly on this workspace. No downloaded pretrained weights or external tokenizer are needed.

The lock records the tested macOS arm64 environment. Linux CPU lab installation must be separately validated and its environment recorded. Install from a reviewed CPU wheel source if avoiding unnecessary accelerator runtime dependencies on Linux; do not treat successful Mac installation as Linux validation.

## Reproduce the engineering evidence

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m aim.reproduce
.venv/bin/python -m aim.comparison_reproduce
```

The second command reproduces phase 1: test log, SFT, preference/DPO, RLVR, a separate trained Judge, an RLVR-from-SFT comparison, reference/trained-Judge evaluations, integrated loop, neural Researcher diagnostic and local benchmark. The third separately reproduces the registered six-model arithmetic-supervision comparison and 1,664 Controller evaluations. Read each run's metrics and failure records. Neither launches distributed processes automatically.

Core tests can run without torch; neural tests explicitly skip. A full reproduction requires torch and numpy and must not be reported successful when those tests are skipped.

## Individual stages

```sh
.venv/bin/python -m aim train --config configs/sft.json
.venv/bin/python -m aim train --config configs/preference.json --initialize runs/<SFT-run>/checkpoint.pt
.venv/bin/python -m aim train --config configs/rlvr.json --initialize runs/<preference-run>/checkpoint.pt
.venv/bin/python -m aim train --config configs/judge.json
.venv/bin/python -m aim loop --judge runs/<judge-run>/checkpoint.pt
.venv/bin/python -m aim evaluate --judge runs/<judge-run>/checkpoint.pt
.venv/bin/python -m aim benchmark
.venv/bin/python -m aim.distributed_launch --ranks 2
```

Replace the angle-bracket paths with actual printed run directories. Checkpoints require their adjacent `.sha256.json` integrity files. `--initialize` starts a new stage with a new optimizer and fixed reference. `--resume` restores the optimizer, reference and torch RNG and requires unchanged data/config except a higher total `steps`. Checkpoint resume has been compared with uninterrupted CPU training at the tensor level.

The local distributed launcher needs loopback socket permission. It runs only on the current host, records launcher/rank logs, and enforces a 90-second launcher deadline. [Hardware instructions](docs/HARDWARE_AND_CLUSTER.md) cover physical-host measurements.

## Scope and limits

- Phase-1 transformer: **90,624** parameters; Phase-2A experimental transformer: **228,096**; independent feature-based Judge: **225**. These are mechanism tests, not final size decisions. The 2M allocation guard belongs to this smoke harness; it is not an AIM scale limit.
- Preference labels shipped here are procedural, not human feedback. The DPO pipeline accepts explicitly attributed external SFT/preference records; actual human collection and annotation review remain pending.
- RLVR is sampled REINFORCE over three exact-arithmetic candidates with a frozen reference. It is not open-ended reasoning RL.
- Judge learning uses proper scoring and validation-only temperature fitting. It is RLCD-style calibration, not a claimed reproduction of an undocumented proprietary algorithm.
- The calibrated Judge fails under a held-out family shift. Confidence never overrides verifier outcomes.
- Retrieval is local lexical matching; arbitrary web retrieval, literature ingestion, symbolic proof engines and arbitrary-code execution are not enabled.
- No large pretraining run, 100M–300M proxy, 1B/7B model or VIT benchmark has been performed.

## Layout and records

| Area | Purpose |
|---|---|
| `aim/` | Interfaces, state, Controller, tools, verifiers, memory, neural models, training/evaluation/benchmark entry points |
| `configs/` | Explicit miniature configurations; larger scale candidates are planning data only |
| `data/` | Local source fixture; generated training datasets are saved inside each run |
| `eval/` | Versioned public engineering suite and SHA-256 sidecar |
| `tests/` | Core, boundary, objective, data and checkpoint tests |
| `docs/` | Architecture, decisions, training/evaluation specifications and next phases |
| `reports/` | Actual results, failure notes and complete changed-file inventory |
| `runs/` | Unique retained experiments, datasets, source snapshots, checkpoints, logs, metrics and failures |

Runs contain code/config/data hashes, Python/package/platform information, seed/configuration, parent checkpoint hashes, state traces and explicit completion/failure. The first runs preceded Git initialization and truthfully record `git_commit: null`; source archives preserve those implementations. Git was subsequently initialized and linked to the owner's AIM repository, so later runs record the actual commit and working-tree status. A completed run can still return UNKNOWN or poor model metrics. Read the result, not only process status.

Runs are excluded from a future Git index to avoid accidentally committing large binary artifacts; they are not deleted. Back them up separately before cleaning or moving this workspace. No model publication license has been chosen; dependency licenses do not automatically establish rights to future training corpora or checkpoints.

## Technical references

- [Architecture and invariants](docs/ARCHITECTURE.md)
- [Separate training objectives and data contracts](docs/TRAINING.md)
- [Evaluation and failure criteria](docs/EVALUATION.md)
- [Hardware and cluster protocol](docs/HARDWARE_AND_CLUSTER.md)
- [Decision record](docs/DECISIONS.md)
- [Implementation phases and unresolved questions](docs/IMPLEMENTATION_PLAN.md)
- [Phase 1 evidence report](reports/phase-1-implementation.md)
- [Structured Researcher guide](docs/STRUCTURED_RESEARCHER.md)
- [Phase 2A evidence report](reports/phase-2a-implementation.md)
- [Plain/worked comparison guide](docs/FINITE_DIFFERENCE_COMPARISON.md)
- [Phase 2A.1 evidence and retained invalid attempt](reports/phase-2a1-implementation.md)
- [Exact legacy checkpoint continuation](reports/research-resume-audit.md)

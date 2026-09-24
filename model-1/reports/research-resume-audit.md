# Researcher continuation audit

Completed 2026-09-25 local time. **All eight equality checks passed.**

The research-specific trainer resumed the actual Phase 2A seed-17 checkpoint at update 1,200, trained its remaining 600 updates on the original training data, and compared the result with the preserved original update-1,800 checkpoint. No new holdout was evaluated.

| Check | Result |
|---|---|
| Model tensors | Exactly equal |
| AdamW optimizer state | Exactly equal |
| Torch RNG | Exactly equal |
| Sampler RNG | Exactly equal |
| Completed update count | Equal, 1,800 |
| Original random-initialization checkpoint hash | Preserved |
| Validation-selected update | Equal, 1,800 |
| Selected validity, prediction agreement and response NLL | Exactly equal |

The audit took 18.89 seconds on the local CPU host. It ran alongside the first seed of the corrected Phase 2A.1 comparison, so neither timing is an isolated throughput benchmark. The exact comparison concerns tensors and metrics, not checkpoint archive bytes, which contain different provenance metadata.

Artifacts:

- [Portable metrics, environment and lineage](phase-2a1-evidence/legacy-resume.json).
- [Executed audit driver](phase-2a1-evidence/legacy-resume-audit.py). From `model-1/`, rerun with `PYTHONPATH=. .venv/bin/python reports/phase-2a1-evidence/legacy-resume-audit.py`. The original Phase 2A run directories must be present.
- Full retained local audit: `runs/20260924T192233-legacy-research-resume-audit-3ec8993d`.
- [Continuation interface and limitations](../docs/STRUCTURED_RESEARCHER.md).

The original checkpoint has no cumulative token counters. The continuation therefore reports `accounting_start_step: 1200`, with 1,166,224 processed positions for the resumed segment only. It does not invent the preceding training cost.

Legacy selection reconstruction checks original sibling checkpoints and sidecars, and cross-checks validation summaries against retained completed-run metrics or, for interrupted runs, prior metric events. New checkpoints also embed the hashed historical selection records. Keep ancestor artifacts; relocation and cross-platform equality remain untested. This audit does not establish distributed checkpoint recovery.

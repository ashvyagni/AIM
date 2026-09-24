# Data strategy: quality, rights, and provenance

## Policy

Publicly downloadable is not equivalent to licensed for model training. Store terms and license at asset/document level, plus jurisdiction, source URL, retrieval time, permitted use, attribution/share-alike/noncommercial restrictions, opt-out/takedown status, and legal review. Default unknown or incompatible rights to retrieval-only or exclude; do not scrape paywalled or access-controlled content. This is an engineering policy, not legal advice.

OpenAlex metadata is documented as CC0, but its license fields describe locations/copies of works and do not make each linked full text CC0. Semantic Scholar’s dataset/API agreement carries use restrictions. arXiv papers have article-specific terms and repository policies; inspect each record. Code needs per-repository/per-file licensing and obligations; avoid assuming GitHub public means permissive. Maintain deletion lineage so a source can be removed and affected datasets/retrains identified.

## Data layers

| Layer | Content | Use / controls |
|---|---|---|
| A. Knowledge | CC/PD reference works, licensed educational sources, permitted OA papers, documentation, tables. | Deduplicate near copies, preserve bibliographic and temporal metadata, reduce boilerplate, balance fields/languages. |
| B. Structured reasoning | Math problems and proofs, code with tests, derivations, protocols, worked examples, formal proofs. | Confirm dataset licenses; preserve problem provenance; create train/validation/test by generator/source family. |
| C. Research process | Query → search → paper reading → extraction → competing hypotheses → experiments → results → conclusion. | Prefer observed, reproducible traces; store exact tool environment and evidence; review human examples. |
| D. Negative/corrective | Invalid proof, code bug, false citation, unsupported claim, confounded experiment, correction. | Pair error with detector and repair; label error type and severity; avoid teaching harmful fabrication as successful target. |
| E. Synthetic environments | Procedurally generated math, causal/scientific worlds, simulated experiments, API/tool worlds. | Ground truth and generator held out by family; randomize superficial templates; publish generator and seeds after evaluation where safe. |

## Construction pipeline

1. Ingest manifests first; enforce allowlist and rights gate before content fetch.
2. Normalize Unicode/layout while retaining original raw object and hash.
3. Detect PII/secrets, malware, spam, low-quality duplication, benchmark contamination and license conflicts.
4. Parse paper structure into abstract/sections/equations/tables/references with page/span offsets and parser confidence.
5. Deduplicate exact, near-duplicate and boilerplate text across splits; use source/group splits to prevent leakage.
6. Tag discipline, genre, date, language, evidence type, license, source reliability and quality; retain uncertain labels.
7. Build mixture experiments rather than assume fixed percentages. Tune using domain-held-out validation loss and downstream trajectory metrics.
8. Track token counts before and after filtering, source concentration, language/domain coverage, rights exclusions and estimated unique-token count.

## Initial mixture hypothesis

Do not lock percentages until pilot data. Start with natural-language licensed knowledge plus high-quality textbooks/reference, software and scientific code, mathematics, scholarly papers, and process trajectories. Oversample high-value reasoning in mid-training rather than letting small scientific corpora dominate base pretraining. Use quality/domain weighting with caps per source and deduplication. Maintain a broad general-language anchor to avoid catastrophic forgetting. Validate exact token allocation by sweep.

## Synthetic data

Generate trajectories from symbolic/math solvers, compilers/tests, theorem provers, simulators and controlled causal worlds. Record the generator executable/version, hidden seed, expected invariants, and independent checker. Sample success, near misses, irrelevant evidence, contradictory sources, broken tools, confounders and underdetermined cases. Prevent train/test family leakage by holding out generator algorithms and latent rules, not just random seeds. Model-generated explanations may be included only when independently validated; teacher confidence is not ground truth.

## Rights tiers

- **Train-ok pending legal review:** explicit permissive license covering intended use, compliance recorded.
- **Conditional:** share-alike, attribution, noncommercial or jurisdictional constraints; do not mix into a model intended for incompatible redistribution until counsel approves.
- **Metadata-only / retrieval-only:** metadata terms permit indexing but full-text/model-training terms are unclear or restricted.
- **Excluded:** unknown, access-controlled, paywalled without license, opt-out, or incompatible terms.

Data lineage must reach final token chunks and checkpoint/run IDs. Publish a datasheet listing categories, counts, filtering and known gaps without disclosing restricted content.

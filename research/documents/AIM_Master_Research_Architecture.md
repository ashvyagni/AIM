# AIM Master Research Architecture

**AIM Research Intelligence Model**  
**Research cut-off:** 24 September 2026  
**Version:** 0.1 — evidence-based proposal, with unresolved hardware-dependent choices  
**Status:** ready for hardware/data audit and falsification experiments; not a final parameter-scale authorization

## Executive decision

Build AIM as a research system rather than a monolithic “scientist” model: an autoregressive researcher, an explicit evidence/research state, a deterministic controller, a small calibrated decision component, independently versioned verifiers, and provenance-aware memory. The distinctive capability to test is whether this system makes *more independently verified, source-supported research progress per unit cost* than a strong retrieval/tool baseline.

Begin with licensed-checkpoint system baselines plus trainable 100M–1B proxies. Use a dense decoder Transformer as the first trainable control. The user reports approximately 70–84 VIT machine-learning-lab systems, each described as 13th-generation Intel Core i9, 32 GB RAM and Intel UHD Graphics 770. If consistent and concurrently available, this implies 2.24–2.688 TB aggregate RAM distributed across hosts, not one addressable memory pool. The fleet may be joinable as a CPU cluster; that capability, network, scheduler, access windows, storage and sustained throughput are unverified. Start with 100M–300M CPU proxies and parallelize data, indexing, verifiers, evaluation, simulation and rollouts. Benchmark 1B distributed scaling before deciding whether a sharded 7B run has a useful schedule. Aggregate memory alone neither proves nor disproves practical 7B training; measured end-to-end throughput, data, budget and cluster reliability determine feasibility. Accelerator access remains preferred for substantial pretraining. Do not choose 14B/32B/70B or a 100B–400B MoE on aspiration alone.

Use SFT → DPO preference baseline → isolated outcome-based RLVR → separately trained/calibrated decision policy → integrated loop only after factorial ablations. Do **not** begin with `w_h R_h + w_v R_v + w_c R_c + w_p R_p`. Correctness, subjective utility, calibrated forecasts and process traces have different semantics and error models. Use hard constraints/release gates for critical correctness/provenance, separate proper-score calibration, and Pareto reporting. Treat the September 2026 RLCD work as an interesting interface/training claim whose public reproducibility is currently insufficient to adopt as an algorithm.

## How to read this proposal

The attached Charter supplies the scientific vision and candidate architecture. The Astra Directive supplies the research protocol and requested artifacts. Both are hypotheses/requirements for investigation; neither is evidence that its claims or scale plan are correct. This report labels established evidence, recommendation, hypothesis, speculation and unknown separately. “Verified” always means a named check passed within a stated scope, not absolute truth.

## 1. Mission and measurable system claim

Target workflow: define a question → identify assumptions and unknowns → retrieve and read evidence → generate competing hypotheses → derive distinct predictions → run computation/experiments/proofs → compare results and contradictions → revise beliefs → write an artifact with claim-level provenance.

Primary outcome proposed for AIM: independently verified supported research progress per unit compute/tool cost. Report its components rather than conceal them in a composite: resolved subtasks, verified claim fraction, citation support, successful/reproducible experiments, correctly rejected hypotheses, contradictions found, calibration of resolved forecasts, unsupported-claim rate, human usefulness, and cost/time. Validate the metric with researchers before optimizing it.

This does not promise discovery of new scientific truths. Scientific novelty, causal validity and future empirical truth often lack immediate machine-checkable labels. Keep these human-reviewed or longitudinal tasks separate from exact verification.

## 2. Architecture choices

### Proposed system: researcher + controller + judge + verifier ensemble

1. **Researcher:** decoder language model proposes structured actions, claims, hypotheses, predictions, analyses and final prose.
2. **Controller:** deterministic shell enforces schemas, permissions, limits, provenance and state transitions. A learned router may rank legal candidate actions.
3. **Decision/judge model:** small separate model initially; emits a typed distribution for explicit tasks such as source-support status, contradiction presence, continue/retrieve/verify/abstain, or hypothesis selection. It cannot declare unobserved claims true.
4. **Verifier ensemble:** code/tests; symbolic math; theorem prover; simulation; statistical recomputation; source/version/span and citation checking; human experts where needed.
5. **Research memory:** relational provenance ledger, evidence graph, object storage, lexical + vector retrieval. Vector similarity is discovery, not proof.
6. **Human review:** for high impact, unresolved conflicts, novelty, invalid/missing source authority or out-of-distribution decisions.

### Alternatives

| Candidate | Pros | Risks | Recommendation |
|---|---|---|---|
| A. Shared backbone + decision head | Efficient, shared knowledge, lower latency | Calibration/objective conflict; head inherits shared errors | Test against separate model; not assumed sufficient |
| B. Researcher + calibrated judge | Independent objective/data/update; modular | Correlated blind spots, latency, judge bottleneck | Best initial integration |
| C. Large MoE + separate judge + verifiers/router | Sparse capacity and role separation | Total memory/optimizer burden, all-to-all, complex failure surfaces | Future only if hardware and proxy studies justify it |
| Retrieval/tool system with existing model | Fastest useful baseline; tests system value | Checkpoint/license dependency; no new base model | Mandatory baseline, not final scientific contribution by itself |

## 3. Foundation model architecture and size

### Scale decision: explicitly unresolved; preliminary 70–84-node lab inventory supplied

The Directive’s dense 7B/14B/32B/70B and MoE 100B–400B+ ranges are candidate design space, not verified feasible targets. VIT’s reported 70–84 CPU systems change the memory and parallel-workload picture, but provide no measured training throughput. Compute estimates below are sensitivity cases; only cluster inventory and representative multi-node benchmarks can establish feasibility. A public frontier report proves specialized infrastructure, not VIT capability.

### Conditional dense baseline

If measured accelerator access permits, profile a conventional decoder-only pre-norm Transformer near 7B: provisional 32 layers, hidden width 4096, 32 query heads, 8 KV heads, dimension 128, SwiGLU, RMSNorm, RoPE, GQA. Exact parameter count must be recalculated from config/vocabulary/embedding tie before run; this shape is near, not guaranteed to equal, 7B. Start at 4K–8K context; 16K/32K staged extension requires explicit retrieval comparison and activation/KV budget.

Tokenizer comparison: byte-fallback BPE or unigram, candidate vocabulary 32K–64K; evaluate domain compression and lossless treatment of Unicode, LaTeX, scientific notation, code, tables and multilingual content. Freeze only after measurement.

Baseline objective: causal next-token NLL. AdamW is the reference optimizer; benchmark Muon under same data, compute and tuning budget before selection. BF16 only where supported and numerically validated; FP32 for CPU proxy; FP8 only on capable accelerator after validation. Use activation checkpointing, fused attention where available, and sharded optimizer/model states as needed.

### Dense vs MoE

Dense activates all weights per token, simplifies load balancing and is robust to weak network. MoE increases total capacity while active compute tracks selected experts approximately; **weights, optimizer states, checkpoint size track total parameters**, while expert token dispatch adds all-to-all traffic. Load imbalance, capacity overflow, router collapse, stragglers and inference residency can erase FLOP savings. DeepSeekMoE’s fine-grained/shared experts are an experimentally supported option, not a universal rule. First test modest 8–16-expert MoE against dense at matched active compute, token count and quality. Track load, route entropy, expert utilization, wall time, communication, cross-domain transfer, inference latency. Do not pin experts to domains initially; specialization/fragmentation is a hypothesis.

### Alternatives in attention/sequence modeling

- **GQA:** strong KV-cache compromise; test quality and cache at same compute.
- **MLA:** DeepSeek reports reduced KV cache; more implementation complexity and kernel dependence; compare end-to-end.
- **RoPE:** portable baseline; YaRN/other extension only after long-context comparisons.
- **FlashAttention-2:** IO-aware exact attention, useful where hardware kernels support it; not a model architecture replacement.
- **State-space/hybrid:** credible alternative for sequence efficiency; defer unless measured attention/context bottleneck and compare on evidence recall/synthesis.
- **Multi-token prediction:** reported by DeepSeek-V3; treat as auxiliary objective ablation, not assumed quality gain.

## 4. Scaling and compute

The standard first-order dense estimate is `F ≈ 6ND`, where `N` is parameters and `D` training tokens. This omits implementation overhead, attention nuance, evaluations, failures, post-training and networking. MoE approximation uses active parameters for token FLOPs but total parameters for memory/optimizer/storage and routing communication. Chinchilla’s compute-optimal findings are a reference, not a universal 20-token law; modern data quality, repeated data and inference demand can move the optimum.

| Candidate | Illustrative token budget | FLOPs `6ND` | Time at 1 PFLOP/s sustained |
|---|---:|---:|---:|
| 7B dense | 140B | 5.88e21 | 68 days |
| 14B dense | 280B | 2.35e22 | 272 days |
| 32B dense | 640B | 1.23e23 | 3.90 years |
| 70B dense | 1.4T | 5.88e23 | 18.7 years |
| 20B-active MoE | 400B | 4.80e22 | 1.52 years |
| 32B-active MoE | 1T | 1.92e23 | 6.09 years |

At 100 TFLOP/s sustained, multiply times by 10; at 10 TFLOP/s, by 100. For the reported 70–84 nodes, illustrative useful per-node throughput scenarios of 10/100/1,000 GFLOP/s correspond to 0.70–0.84/7.0–8.4/70–84 TFLOP/s aggregate and roughly 222–266/22.2–26.6/2.22–2.66 years for the 7B run, assuming perfect linear scaling. These are not measured estimates and omit communication, downtime, utilization and other overhead; actual schedule would be longer. Sensitivity is linear in token count and inverse in sustained throughput. Aggregate RAM is not pooled.

BF16 weights require `2N` bytes: 7B≈14GB, 14B≈28GB, 32B≈64GB, 70B≈140GB, MoE 300B≈600GB. A conservative 16B/parameter Adam training-state estimate gives 7B≈112GB, 14B≈224GB, 32B≈512GB, 70B≈1.12TB, 300B≈4.8TB before activations/workspaces. Sharding distributes these bytes but does not remove them. KV cache for standard K/V ≈ `2 × layers × sequence × KV_heads × head_dim × bytes`; a 32-layer, 8-KV-head, 128-dimension BF16 model at 32K is roughly 4 GiB per sequence before overhead.

**Scale recommendation:** 100M–300M single-node CPU proxies first; use the potential fleet for parallel data/verifier/evaluation work immediately, then measure a distributed 1B pilot. Gate sharded 7B experiments on measured scaling, schedule, data and budget; dedicated accelerators are preferred for substantial pretraining. 14B/32B only after scaling curves, licensed tokens, run allocation, storage and inference plan justify them; 70B/MoE stretch only. Full calculation and assumptions: [scaling report](../research/scaling/scaling-and-compute-analysis.md) and [updated VIT hardware audit](../research/hardware/vit-ml-lab-audit.md).

## 5. Data program

Build licensed, versioned corpus layers:

- **A Knowledge:** permitted reference/educational material, open scholarly full texts only when their specific rights permit, technical documentation.
- **B Structured reasoning:** math, proofs, code/tests, derivations, experiments and protocols.
- **C Research process:** question/search/evidence/hypothesis/test/result/conclusion traces.
- **D Negative/corrective:** wrong proof/citation/statistic/experiment paired with diagnosis and repair.
- **E Synthetic environments:** math, causal and scientific worlds with exact hidden ground truth and randomized generators.

Access does not establish model-training rights. OpenAlex metadata are CC0 but linked work/full-text rights can differ; Semantic Scholar API has its own agreement. Track rights at source/object level, including attribution/share-alike/noncommercial/opt-out; default unknown rights to exclude or retrieval-only pending legal review. Record manifest, hash, transformations, parser, license, domain/date, dedup cluster, contamination lineage and resulting runs. Hold out entire source families, dates and synthetic generator families.

Do not set final mixture percentages before inventory and pilot. Compare broad high-quality corpus and reasoning-heavy continued-pretraining mixtures at equal compute while preserving general-language anchor. Synthetic teachers are not labels: accept only outputs checked by independent tools or expert adjudication.

## 6. Training: RLHF, RLVR, RLCD, process feedback

### Signals and math

- **LM:** `-E Σ_t log πθ(x_t|x<t)`; models text distribution, not truth.
- **Preferences/RLHF:** pairwise Bradley–Terry `P(yw>yl)=σ(rφ(x,yw)-rφ(x,yl))`; PPO optimizes clipped policy objective with KL to reference. DPO is a simpler offline preference baseline under its derivation assumptions.
- **RLVR:** objective outcome reward from trusted math/code/symbolic/proof/simulation/statistics checks; compare REINFORCE/PPO and GRPO. GRPO uses within-prompt relative rewards and has normalization/length/difficulty effects; DAPO/Dr. GRPO variants require controlled reproduction.
- **RLCD-style decisions:** typed action/outcome distribution and calibrated scores for defined, resolved tasks. Optimize Brier or log score; evaluate on held-out domains/shifts. Abstention is an action chosen from expected task costs.
- **Process reward:** step label or value estimate may guide search; it is not proof. PRM evidence is strongest in math and a particular best-of-N setup. Keep it auxiliary/bounded and evaluate plausible wrong traces.

Brier `(p-y)^2` and log loss are proper scoring rules under standard assumptions; ECE is a bin-dependent diagnostic, not a proper loss. Calibration is for a specified event/reference class/horizon. “Claim is entailed by these spans” is different from “hypothesis will replicate next year.” Unresolved scientific truth is censored/unresolved, not a fabricated binary label.

### Recommended training order and controls

1. Base pretrain and scaling proxies; 2. mid-train science/math/code and context only if measured; 3. Research SFT for structured actions/evidence and communication; 4. DPO preference baseline; 5. isolated verifier-grounded RLVR; 6. separately trained decision policy with supervised proper score then RL only where action returns are observable; 7. optional process reward; 8. integrated state/action RL study.

This order is a pragmatic experiment, not a claim sequential is optimal. Compare interleaved/joint methods after components work. Avoid a single scalar sum: make source/provenance and critical verification release constraints; separately optimize preference and decision calibration; report Pareto front. If reward scalarization is tested, normalize using frozen training statistics, specify weights, perform sensitivity analysis and monitor independent outcomes. Stop on proxy reward/outcome divergence, reward hacking or held-out regression.

## 7. Research agent, memory, verifiers

The `ResearchState` contains question/scope, assumptions, facts/unknowns, sources, evidence spans, hypotheses/predictions, experiments, results, contradictions, forecasts/confidence, open tasks, provenance and final claims. Represent state as append-only events with supersedes/retracts edges and a materialized relational/graph view. Store originals/results as immutable objects; lexical/vector indices retrieve candidates. Each claim links source version → exact evidence span → derivation/transformation → verifier result → confidence target.

Actions: SEARCH, READ, EXTRACT, COMPARE, CALCULATE, CODE, RUN, SIMULATE, PROVE, CHALLENGE, HYPOTHESIZE, VERIFY, ABSTAIN, REVISE, WRITE. Each has schema, allowed resources, timeout, cost, result type and error state. Deterministic shell enforces contracts; learned researcher/router proposes actions. For candidate action use expected information value, impact and cost; stop at marginal value below cost, hard budget, or human-review boundary.

Verifier stack: schema and source identity; code compile/tests/property tests in fresh sandbox; SymPy/numeric checks with assumptions/tolerances; Lean/Isabelle proof where formalization makes sense; simulation/statistical recomputation; citation entailment and contradiction checks; independent small judge; human review. Preserve raw output/version/hash. “VERIFIED” means that specific test passed on that artifact and scope. A citation is not evidence unless the cited span supports the claim.

## 8. Failure modes and defenses

| Failure | Defense / measurement |
|---|---|
| Reward-model overoptimization, verbosity/style gaming | DPO/PPO comparison; hidden human audit; length-controlled pairs; independent outcomes; KL and early stop |
| Verifier loopholes / test leakage | Fresh sandbox; hidden/rotating tests; independent implementations; metamorphic testing; red-team; grader audits |
| Confidently wrong judge | Proper-score held-out evaluation, domain/time shift, independent verifier ensemble, abstain/review, human audit |
| Fabricated/weak citations | Resolve source/version, exact span and entailment; unsupported-claim metric; source quality and contradiction review |
| Calibration collapse under distribution shift | Per-domain/time/source curves, Brier/log, selective risk–coverage, OOD review gates |
| Process reward values plausible invalid steps | Outcome-conditioned labels, terminal verifier, adversarial near-miss cases, bounded use |
| MoE router imbalance / fragmentation | All-to-all/load metrics, cross-domain transfer, matched dense baseline; defer if not measured viable |
| Context loses evidence / retrieval bias | Exact provenance, hybrid search, retrieval ablation, evidence recall, claim graph, contradiction injection |
| Data contamination/rights failures | Rights manifest, dedup, sealed sets, source lineage, exclusion defaults and removal process |
| Scientific delay/no ground truth | Forecast ledger with target/horizon/resolution; unresolved/censored state; expert longitudinal review |
| Demo/research theater | Preregistered evaluation, negative runs, independent reproduction, open configs/artifacts where permitted |

## 9. Evaluation framework

Create sealed, source-family/generator/time-held-out suite before training. Measure knowledge and temporal facts; math exact answers, derivations and proof validity; code hidden tests/complexity/security; scientific hypotheses and experimental design; literature synthesis/citation support/contradiction discovery; calibration and selective action; multi-step task completion, state retention, provenance and cost; scholarly communication by blind experts. Cover math, CS, physics, quantum information, astronomy, chemistry, biology, psychology/cognitive science, statistics, engineering, economics, philosophy/logic, research methodology and writing. Report each domain separately.

Public diagnostic anchors: GPQA, MATH/FrontierMath when licensed/available, Lean tasks, SWE-bench variants, SciCode, ALCE, PaperBench and long-context retrieval benchmarks. Public scores are vulnerable to contamination and are not an overall research-intelligence metric. LLM graders require human validation, blinding, verbosity/position tests and inter-rater analysis. Primary measures: claim support precision/recall, verified result correctness, unsupported-claim rate, Brier/log score, risk-coverage and verified research progress/cost. Use confidence intervals, multiple seeds where possible, budget-matched baselines and report all failures.

## 10. Hardware and implementation plan

### Audit first

Inventory CPU SKU/instructions/cores/NUMA/RAM/storage, GPU/VRAM/runtime, job limits, scheduler, uptime, network topology/RDMA/bandwidth/latency, quotas. Benchmark representative forward/backward/update, tokenizer/decompression, inference at 1K–32K, all-reduce and MoE all-to-all, checkpoint/resume, full-day availability. Record p50/p95, utilization and repeats. The user reports ~70–84 VIT lab systems, each described as 13th-generation Intel Core i9, 32 GB RAM and UHD Graphics 770. This preliminary inventory implies 2.24–2.688 TB aggregate installed RAM if consistent, distributed across nodes. It does not establish SKU consistency, simultaneous access, schedulable cluster status, network/storage performance, throughput or additional GPUs. Inventory and benchmark each node class; measure representative single-node and multi-node training, inference, all-reduce, checkpointing and availability. The Codex host separately reported Apple M3 integrated GPU (10 cores); it is not part of the VIT estimate.

If GPU nodes are present: dense DP/FSDP first; add TP/PP/context parallel only as memory/topology require; EP only after MoE test. CPU nodes provide valuable corpus preparation, dedup, search, verifier execution, theorem/symbolic work, simulation, reward generation, evaluation and rollout orchestration even if they cannot economically pretrain final scale. Actual contribution depends on benchmark and job constraints.

## 11. Milestones and research value

**0.** Lab access + hardware benchmark + rights audit.  
**1.** Data manifest, tokenizer studies, evidence graph, verifiers, baseline retrieval-agent.  
**2.** 100M/300M/1B dense scaling/objective proxies.  
**3.** SFT/preferences/RLVR/RLCD factorial post-training.  
**4.** Long-horizon loop with contradiction, tool failure and held-out source tests.  
**5.** Choose 7B/14B/32B only from measured scaling/budget; fund accelerator access if needed.  
**6.** External replication; publish results and limitations. 

An AIM contribution is scientifically meaningful if it shows reproducible causal gains over a strong retrieval/tool and same-model baseline at fixed budget, with component ablations, independent graders, held-out domains/generators, and no unacceptable calibration/factuality regression. “We used RLHF + RLVR + RLCD” or “we trained a large MoE” is not sufficient.

## 12. References and claim anchors

1. Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), 2022.
2. Rafailov et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290), 2023.
3. Shao et al., [DeepSeekMath](https://arxiv.org/abs/2402.03300), 2024.
4. Guo et al., [DeepSeek-R1](https://arxiv.org/abs/2501.12948), 2025.
5. Yu et al., [DAPO](https://arxiv.org/abs/2503.14476), 2025; [Dr. GRPO analysis](https://arxiv.org/abs/2503.20783), 2025.
6. Lightman et al., [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050), 2023; [PRM800K data/code](https://github.com/openai/prm800k).
7. Gao et al., [ALCE](https://arxiv.org/abs/2305.14627), EMNLP 2023.
8. Dai et al., [DeepSeekMoE](https://arxiv.org/abs/2401.06066), 2024; DeepSeek-AI, [DeepSeek-V3](https://arxiv.org/abs/2412.19437), 2024.
9. Yang et al., [Qwen3](https://arxiv.org/abs/2505.09388), 2025; GLM-4.5, [technical report](https://arxiv.org/abs/2508.06471), 2025; Team OLMo, [OLMo 3](https://arxiv.org/abs/2512.13961), 2025.
10. Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), 2022; Kaplan et al., [Scaling Laws](https://arxiv.org/abs/2001.08361), 2020; Sardana et al., [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448), 2024.
11. Liu et al., [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982), 2025.
12. Ainslie et al., [GQA](https://arxiv.org/abs/2305.13245), 2023; Dao, [FlashAttention-2](https://arxiv.org/abs/2307.08691), 2023/ICLR 2024; Peng et al., [YaRN](https://arxiv.org/abs/2309.00071), 2023.
13. Guo et al., [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599), 2017; Kadavath et al., [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221), 2022; [LLMs Must Be Taught to Know What They Don't Know](https://arxiv.org/abs/2406.08391), NeurIPS 2024.
14. TypeSafe AI, [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), 15 Sep 2026; [RLCD description](https://www.jevtypesafeai.com/jev/rlcd). Company announcement; reproducible method not disclosed in reviewed sources.
15. Yao et al., [ReAct](https://arxiv.org/abs/2210.03629), 2022; Schick et al., [Toolformer](https://arxiv.org/abs/2302.04761), 2023.
16. Rein et al., [GPQA](https://arxiv.org/abs/2311.12022), 2023; Starace et al., [PaperBench](https://arxiv.org/abs/2504.01848), 2025.
17. [Megatron Core parallelism documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html); [OpenAlex API and license docs](https://help.openalex.org/api/), [Semantic Scholar data license](https://api.semanticscholar.org/license/).

See the supporting literature matrix, architecture/algorithm reviews, hardware report, data strategy and decision record for expanded comparisons, source quality and unresolved questions.

# Open research questions and falsification plan

| Question / hypothesis | Current status | Minimum experiment that can falsify it | Decision consequence |
|---|---|---|---|
| RLHF + RLVR + calibrated decisions improve research outcomes jointly. | Hypothesis, not demonstrated for AIM. | 2×2×2 factorial post-training on same 1B proxy/data; sealed multi-step tasks; fixed token/tool budget, 3 seeds. | Drop any component without independent gain or if it harms core metrics. |
| Calibrated decision model is better separate than a head. | Unknown. | Shared head vs separate judge with same training examples and inference budget; domain/time-shift Brier, log score, selective utility. | Choose simplest configuration on Pareto frontier. |
| Structured provenance state reduces unsupported claims and forgetting. | Evidence-backed motivation; AIM effect unknown. | Transcript-only vs append-only evidence graph on matched long tasks; blind adjudication and injected contradiction. | Build only if traceability and success gains outweigh latency. |
| Process reward helps beyond outcome verification. | Mixed/narrow evidence. | PRM vs no PRM on proof and simulation environments; hold out step patterns; probe plausible wrong steps and verbosity. | Keep only bounded search/critique role if robust. |
| MoE improves quality per FLOP without harmful domain fragmentation. | Plausible, not shown for research. | Dense and modest MoE proxy, same active FLOPs/tokens; transfer, expert utilization, all-to-all, wall-clock. | Require clear operational advantage before scale-up. |
| Native long context is preferable to retrieval/chunking. | Unknown and workload-dependent. | Matched evidence tasks, context lengths, retrieval; score exact span recall, synthesis, cost and latency. | Train longer only where measured net improvement. |
| Better pretraining data beats larger parameter count at project budget. | Strong scaling-law motivation; project-specific unknown. | Nested proxy runs with quality-mixture vs broad corpus at equal compute. | Use loss+research task Pareto to allocate budget. |
| CPU cluster can deliver more than infrastructure support. | Likely for preprocessing/verifiers; throughput unknown. | Audit plus 24h proxy benchmarks and job cost accounting. | Decide local jobs vs accelerator request/cloud by workload. |
| System learns to update beliefs rather than rationalize initial answer. | Hypothesis. | Hidden sequential evidence tasks with randomized order, contradiction injection, scoring for appropriate revision and preserved history. | Make belief revision primary outcome if successful. |
| Research-loop RL is a novel contribution. | Speculative. | Compare learned policy to strong fixed/retrieval baselines; independent replication and pre-registration. | Claim novelty only if causal gains generalize and are not scaffold artifacts. |

## Unresolved empirical and governance questions

- What accelerator type, quantity and reserved hours are actually available at VIT? What are cluster access and data rules?
- How many unique, explicitly reusable tokens can be assembled across scientific and educational domains?
- Which primary endpoint do domain experts consider useful and reliable enough to guide research-system development?
- Can support/calibration labels be collected consistently across disciplines, and how should disagreement be represented?
- How much calibration transfers between a model’s own claims, claims grounded in supplied sources, and future outcomes?
- What is the response to source retraction, version correction, and evidence graph deletion?
- What security envelope is needed before tool actions can run code or access network resources?
- Which benchmark families remain unexposed and licensed for public evaluation?
- Can model-assisted grading be calibrated sufficiently for large-scale science-task rubrics, or is expert review the bottleneck?
- What is a realistic objective for novelty? Novelty detectors are retrieval heuristics, not proof of new knowledge.

## Evidence needed before publication

Preregistered hypotheses and exclusions; benchmark/data rights; sealed test suite; runs at matched budget; independent verifier audit; baseline and ablations; negative results; confidence intervals; reproducible configs/environment; external replication; narrowly worded claims. “Research intelligence” or “scientific discovery” must be supported by trajectory outcomes, not a demo or benchmark composite alone.

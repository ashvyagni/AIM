# RLHF + RLVR + RLCD: review and proposed integration

## Main conclusion

These labels describe unlike supervision. RLHF targets human-valued quality/preferences; RLVR targets task outcomes that a trusted program can check; RLCD, as publicly described by TypeSafe in September 2026, targets structured probabilistic decisions. A single weighted scalar can obscure failures and let a high prose-preference score compensate for an incorrect result. AIM should begin with separated datasets, heads/policies where useful, and constrained gates; only combine rewards after measuring scale, noise, and tradeoffs.

The Jev/RLCD announcement is a company claim. Its public materials describe typed decisions and a probability distribution/confidence; the “zero hallucination” wording is expressly about output type/range conformance. The public material reviewed does not disclose a reproducible loss, RL algorithm, training data, or independent calibration results. Thus RLCD here means an AIM research objective family, not an implementation copied from Jev.

## Mathematical objects

Autoregressive pretraining minimizes token negative log-likelihood:
\[
\mathcal L_{LM}(\theta)=-\mathbb E_{x\sim D}\sum_t m_t\log\pi_\theta(x_t\mid x_{<t}),
\]
where `m_t` masks padding or excluded spans. This is not a direct factuality objective.

For pairwise preferences, Bradley–Terry models `P(y_w ≻ y_l | x)=σ(r_φ(x,y_w)-r_φ(x,y_l))`, optimized with binary cross entropy. The conventional KL-regularized target is `max_π E[r(x,y)] - β KL(π||π_ref)`. PPO clips the probability ratio `ρ_t(θ)=π_θ(a_t|s_t)/π_old(a_t|s_t)` in the surrogate `E[min(ρ_t A_t, clip(ρ_t,1-ε,1+ε)A_t)]`, with value/GAE and KL controls in common implementations ([PPO](https://arxiv.org/abs/1707.06347)). RLOO is a REINFORCE-style leave-one-out baseline that can avoid a separate value/critic model ([Ahmadian et al.](https://arxiv.org/abs/2402.14740)); compare it where critic memory is material. DPO eliminates the separate online RL loop under its assumptions; it is an excellent low-cost preference baseline, not proof that RL is unnecessary for sequential environment interaction.

Preference-algorithm comparison for AIM: **DPO** is a clean pairwise/reference-policy baseline; **IPO** changes the preference objective to address issues in DPO's assumptions and should be evaluated where label noise/ties matter ([Azar et al.](https://arxiv.org/abs/2310.12036)); **KTO** uses binary desirable/undesirable feedback and a prospect-theoretic utility, useful when pairwise labeling is expensive but it changes assumptions about human utility ([Ethayarajh et al.](https://arxiv.org/abs/2402.01306)); **ORPO** integrates an odds-ratio penalty into SFT and removes a separate reference model, reducing orchestration but coupling instruction imitation and preference learning ([Hong et al.](https://arxiv.org/abs/2403.07691)); **SimPO** uses length-normalized sequence log probability as an implicit reward with a target margin, avoiding a reference model ([Meng et al.](https://arxiv.org/abs/2405.14734)). **PPO-RLHF** enables on-policy exploration through an explicit learned RM but costs rollout/critic infrastructure and is vulnerable to RM overoptimization. **RLAIF/Constitutional AI** can scale critique/preferences but inherits evaluator bias and should be validated against human/domain labels ([Bai et al.](https://arxiv.org/abs/2212.08073)). **Self-rewarding** systems risk circular teacher/student errors; use only with external objective checks. AIM should implement DPO first, then one alternative at a time on the same preference set; select by blind research utility and factuality, not headline chat leaderboard.

For RLVR, an episode `τ` receives a terminal or verifiable reward such as `r_v(τ)∈{0,1}` from a grader, theorem prover, tests, or recomputation. REINFORCE uses `(r-b)∇ log π(τ)`. PPO applies clipped token/trajectory ratios. GRPO samples a group for a prompt and centers/scales outcome rewards, often `A_i=(r_i-mean(r))/(std(r)+ε)`; this removes a learned value model but creates group-relative and difficulty/length effects. DAPO changes sampling/clipping/normalization choices; Dr. GRPO analyzes normalization bias. These are algorithm variants to ablate, not guaranteed improvements.

For pairwise choice probabilities `p`, Brier loss is `(p-y)^2` (multiclass `Σ_k(p_k-y_k)^2`) and log loss is `-y log p -(1-y)log(1-p)`. They are strictly proper under standard assumptions: expected loss is optimized by reporting the true conditional distribution. Brier is bounded but decomposes discrimination and reliability; log loss punishes confident errors strongly. Use both. ECE bins confidence and compares mean confidence/accuracy, but is bin-dependent, sample-hungry and not a proper scoring rule; report confidence intervals, adaptive bins, reliability curves and proper scores. Accuracy alone cannot measure calibration.

Selective prediction with action `abstain` and cost `c_a` should be trained/evaluated against actual operational costs. For a calibrated binary event with action loss `L(a,y)`, choose `a*=argmin_a E[L(a,Y)|x]`; abstain when its expected loss is lower. Thresholds are product costs, not a universal magic confidence. Track risk versus coverage, utility under preregistered costs, and whether abstention is useful (not blanket refusal).

## Why scalar reward is risky

Suppose outcome quality has vector `R=(R_h,R_v,R_c,R_p)`: human usefulness, verified correctness, calibration/decision quality and process quality. Scalarization `wᵀR` chooses a tradeoff only if scales and weights represent real exchange rates. If `R_v=0` (fatal fact/error) but `R_h` is high, unconstrained sum may still reward it. Prefer a hierarchy / constraints for critical correctness:
\[
\max_\pi\ E[R_h+\lambda R_{progress}] -\beta KL(\pi||\pi_{ref})
\quad\text{s.t. }E[1-R_v]\le\epsilon_v,\ E[\text{unsupported claim rate}]\le\epsilon_u,
\]
with calibration optimized and reported separately. If constraint optimization is unstable, use lexicographic selection: verifier pass → supported claims → calibrated control decisions → preference quality, with a Pareto dashboard. The calibration score applies to decisions with observable outcomes, not to all prose assertions at once.

## Proposed staged experiment, not final recipe

1. **Research SFT:** train output/action formats and correct tool use from curated demonstrations. Keep raw sources and provenance; do not label generated chain-of-thought as ground truth by default.
2. **Preference baseline:** DPO on human pairwise comparisons over answer usefulness, clarity, relevance, and faithful reporting. Maintain separate dimensions and expert disagreement; sample for human adjudication. Compare PPO+learned RM only if online interaction/data justify complexity.
3. **RLVR islands:** separate math, code, symbolic and simulation environments with trusted execution, hidden tests, timeouts, sandboxing and independent test generation. Use outcome reward first. Compare PPO vs GRPO at same sample and accelerator budget.
4. **Calibrated decision policy:** take immutable research-state snapshots; emit typed distributions for claim state, verify/continue/retrieve/abstain and candidate action ranking. Train with proper scoring losses on adjudicated outcomes and logged environment returns. Start offline supervised calibration; test RL only where action consequences yield clean delayed rewards. Keep decision calibration metrics outside language-policy scalar reward initially.
5. **Process signal:** evaluate PRM as candidate-step ranking / critique auxiliary trained on step labels and outcomes. It must not directly certify truth. Use outcome-conditioned process targets, counterfactual invalid-step pairs, and terminal verification; compare against no-PRM.
6. **Integrated research-loop RL:** state/action sequence `s_t,a_t,o_{t+1}`, rewards for verified evidence, successful experiments, supported final claims and cost. Use deterministic environment contracts. Run factorial ablations: no RL; DPO only; RLVR only; decision policy only; combinations; process reward; shared vs separate judge.

## Delayed and unobservable truth

For scientific claims with no immediate ground truth, do not fabricate labels by forcing a binary truth score. Calibrate only an explicitly defined target, horizon and reference class, e.g., “does this claim follow from these provided sources?” versus “will this hypothesis replicate in a future experiment?” The former has a present, auditable label; the latter may require longitudinal outcomes and censoring. Store forecast issue time, resolution rule, observation horizon, and unresolved status. Use proper scoring only after outcome resolution; treat missing resolution as censored, not incorrect or correct. For causal/hypothesis novelty, human expert review and future studies are needed.

## Adversarial answers

1. Does the trio improve research ability? **Unknown**; only controlled ablations answer it.
2. Calibration without outcomes? Only proxy/reference-class calibration; cannot empirically establish unknown future truth.
3. Delayed outcomes? Forecast ledger, fixed resolution criteria, time-to-event/censoring, periodic follow-up.
4. Can judge supervise generator? Yes for action choice/selection if labels are valid, but it cannot prove unseen prose true; maintain independent auditing.
5. Judge abstains when expected decision cost exceeds abstention cost, out-of-support input, conflicting verifiers, missing provenance, or low effective sample support.
6. Prevent confidently wrong judge: held-out domains/time periods, independent judges/verifiers, adversarial positives/negatives, calibration confidence intervals, distribution-shift detector, human audit.
7. Stop verifier gaming: hidden tests, metamorphic/property tests, randomization, sandbox, exploit red-team, verifier versioning and audits.
8. Prevent verbosity reward: length-normalized preference comparisons, claim-level utility, cost penalty, blinded length controls, no reward for visible reasoning volume.
9. Process reward may guide search but risks rewarding plausible style; test on novel tasks and wrong-but-plausible traces. Include the outcome-guided process-reward study cited in the Directive ([Rezaei et al., 2026](https://arxiv.org/abs/2604.02341)) alongside PRM800K; verify scope and reproduce before adopting.
10. MoE fragmentation: not established as a general research risk/benefit; measure cross-domain transfer, route entropy, expert mutual information and ablation with same active compute.
11–13. Pretraining, data and scale bottleneck are open; estimate by nested ablations. Likely bottleneck is not simply parameters: high-quality independent feedback, verifiers and compute matter.
14. CPU clusters strongly help data, search, tests, synthesis, evaluation and CPU-native scientific jobs; contribution to full pretraining depends on measured sustained throughput and cost.
15. New contribution requires preregistered ablation showing improved trajectory-level, independently graded research outcomes per compute, robustly across held-out domains, beyond retrieval/tool and same-model baselines.

# Model-1 training specification

## Status and scope

The implemented progression is SFT → preference/DPO → verifier-grounded RLVR, with a separately trained Judge and a runtime integration of that Judge. The reproduction also runs SFT → RLVR as a comparison. Integrated research-loop policy training is deferred. The transformer has 90,624 parameters and the five-feature Judge has 225; both start from local random weights.

Using PyTorch numerical kernels is compatible with training from scratch. No pretrained model or pretrained tokenizer is loaded. This phase does not constitute large-corpus language pretraining. The byte tokenizer is a temporary transparent engineering choice.

## Datasets

### Included procedural fixtures

`aim-procedural-v1` generates integer addition examples and polynomial measurement tasks. Addition rows preserve operands, operation, prompt, response, chosen/rejected answers, group, split, generator and rights declaration. The first 64 selected training rows and 32 validation rows are used. Selection follows deterministic operand-pair hash buckets; this tiny subset is not a representative arithmetic benchmark. Labels are explicitly marked procedural and not human feedback.

Judge fixtures comprise 256 training, 96 validation, 96 in-family test and 96 cubic out-of-family records. Whole coefficient groups are disjoint across splits. Multiple candidate forecasts share a world, so candidate rows are correlated; confidence intervals in future studies must resample whole worlds. Features use observations available before the target measurement, never the target outcome. Exact rational arithmetic generates outcome labels independently of the floating-point hypothesis fit.

All examples are project-generated fixtures; no scientific literature or scraped corpus was used for these runs. The rights field documents fixture origin, not a legal review of any future external data.

### External SFT/preference intake

Set `dataset_path` in a new config to a local JSON file with:

```json
{
  "schema": "aim-supervised-v1",
  "version": "team-reviewed-dataset-v1",
  "train": [
    {
      "id": "example-1",
      "group": "task-family-1",
      "prompt": "Example training prompt",
      "response": "Supervised target",
      "chosen": "Preferred response",
      "rejected": "Less preferred response",
      "rights": "Record the actual permission or license",
      "label_origin": "synthetic"
    }
  ],
  "validation": [
    {
      "id": "example-2",
      "group": "task-family-2",
      "prompt": "Different validation prompt",
      "response": "Supervised target",
      "chosen": "Preferred response",
      "rejected": "Less preferred response",
      "rights": "Record the actual permission or license",
      "label_origin": "synthetic"
    }
  ]
}
```

This schema example is illustrative, not a useful training dataset. `label_origin` must be human, synthetic or programmatic. Human records require an `annotation_batch` identifier. The loader rejects missing provenance, duplicate IDs, overlapping groups, exact prompt leakage, identical chosen/rejected responses and conflicting split declarations. It cannot independently establish whether a claimed human annotation was actually produced by a person; annotation audit remains necessary. Near-duplicate/semantic leakage detection is a future ingestion requirement.

Actual RLHF data collection must use blinded response order, task/rubric/version records, independent annotators, disagreement tracking, adjudication and data-use permission. Do not mark the included procedural preference experiment as a human RLHF result. No preference reward model or PPO trainer is implemented in this phase.

External RLVR intake deliberately fails until a dataset-specific verifier adapter is implemented and reviewed. Unknown verifier outcomes must remain missing labels, not automatic failures or successes.

## Objective 1: response-masked SFT

For prompt x and target response y, including EOS:

\[
L_{SFT}=-\frac{\sum_{i,t\in response_i}\log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})}{\sum_i |response_i|}.
\]

Prompt and right-padding positions have zero loss. A BOS token anchors the first prompt. Context overflow fails explicitly; there is no silent truncation. Tests check response token counts, padding independence and causal prefix invariance.

Delivered config: 60 steps, batch 8, AdamW learning rate 0.002, weight decay 0.01, seed 17, global gradient norm clipped at 1.0. FP32 CPU execution uses one torch thread and deterministic algorithms. These are smoke settings, not optimized large-model hyperparameters.

## Objective 2: DPO preference baseline

For preference pair (x,y+,y−), let sθ(y|x) be the sum of response-token log probabilities. A frozen copy of the stage input is the reference πref:

\[
L_{DPO}=-\mathbb E\log\sigma\left(\beta[(s_\theta(y^+|x)-s_\theta(y^-|x))-(s_{ref}(y^+|x)-s_{ref}(y^-|x))]\right).
\]

DPO directly optimizes pairwise preferences without a separately fitted scalar reward model. This is the preference baseline selected for this phase; it is not an implementation of PPO-based RLHF. See [Rafailov et al., Direct Preference Optimization](https://arxiv.org/abs/2305.18290).

Delivered config: 30 steps, batch 4, learning rate 0.0003, β=0.1, fixed SFT reference. Chosen/rejected log-probabilities are length-sensitive sequence sums, consistent with the defined loss; future datasets must evaluate length bias explicitly. Tests verify the zero-margin loss is log(2), preferred gradients have the correct sign, and the reference remains fixed across updates.

## Objective 3: bounded RLVR

The environment offers three candidate integer answers C(x) = {n−1,n,n+1}, where n is computed by the fixture environment. Therefore the answer is guaranteed to be present. This intentionally simplified action space is an important source of privilege and prevents interpreting its metrics as unrestricted answer-generation accuracy.

For candidate score sθ(a|x):

\[
p_\theta(a|x,C)=\frac{\exp s_\theta(a|x)}{\sum_{b\in C}\exp s_\theta(b|x)}.
\]

The verifier independently computes exact arithmetic from operands and returns r(a)∈{0,1}; it does not inspect the preference label. One action is sampled from the current categorical policy. The baseline is the detached exact expectation b=Σa pθ(a)r(a) over this small action space. The stochastic loss is:

\[
L_{RLVR}=-(r(a)-\operatorname{stopgrad}(b))\log p_\theta(a)
 + \lambda\sum_{c\in C}p_\theta(c)\log\frac{p_\theta(c)}{p_{ref}(c)}.
\]

The first term is a REINFORCE policy-gradient estimator for this finite candidate task; the second is its exact candidate-distribution KL penalty. This KL is **not** the KL of the full language-model distribution. The reference is detached. Reward calls, sampled reward, entropy, candidate KL, loss and gradient norm are logged separately.

Delivered config: 30 steps, batch 4, learning rate 0.0003, λ=0.02. The reference is the selected input checkpoint. Reproduction compares a DPO-initialized run with an SFT-initialized run. No PPO clipping, GRPO, learned critic, process reward, long-horizon credit assignment or free-form rollout engine is claimed.

The two branches have different total prior training budgets. Their single-seed results are diagnostics, not a controlled proof that DPO helps or harms RLVR. A future ablation must match total examples, updates, verifier calls and compute across seeds.

## Objective 4: separate calibrated Judge

The defined event z=1 is: “this candidate prediction matches the next synthetic measurement within absolute tolerance 10⁻⁸.” For pre-measurement features f and separate weights φ, qφ=σ(gφ(f)). Implemented proper scoring options are:

\[
L_{log}=-[z\log q_\phi+(1-z)\log(1-q_\phi)],\qquad
L_{Brier}=(q_\phi-z)^2.
\]

The default trains log loss for 150 steps, batch 32, learning rate 0.01. Afterwards choose T from the declared grid {0.5,0.75,1,1.25,1.5,2,3,5} by validation log loss, and use q=σ(gφ/T). Neither test nor OOD labels select T. This is a deliberately small grid-search implementation of temperature scaling, whose empirical calibration role is discussed in [Guo et al., On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).

The experiment demonstrates that temperature fitting on familiar data can worsen shifted-family confidence. It supplies no guarantee of calibration on scientific research claims. The Judge may ABSTAIN below a configured threshold; the default threshold 0 requests verification for all candidates. Runtime verification remains mandatory even for high-confidence predictions.

This is **RLCD-style decision calibration by supervised proper scoring**. It is not reinforcement learning on a sequential decision MDP, nor a reproduction of proprietary RLCD training. To move to decision-policy RL, first define actions, observable state, time-dependent outcomes, costs, delayed feedback and off-policy evaluation. Keep that proposed method in a new experiment/decision record.

## Stage transitions, checkpointing and failure handling

`--initialize` loads only compatible AIM-owned LM weights, creates a new optimizer and freezes a new reference. `--resume` instead checks stage/config/dataset identity, restores optimizer/reference/torch RNG and advances from the recorded step. The deterministic batch order follows step and seed. Every checkpoint has a content-hash sidecar. The user-provided path, parent hash and exported dataset accompany the run.

There is no current random Python/numpy sampling in the update loop; those libraries are seeded, and procedural generation is deterministic. A future stochastic dataloader must preserve its own RNG/cursor and worker state before claiming exact resume. Cross-device and cross-library-version bitwise continuation is not promised.

Nonfinite loss/gradient, incompatible checkpoint, invalid data, excessive allocation or illegal resume fails with retained traceback and configuration. Unknown research outcomes are not discarded or turned into incorrect numeric labels.

## Integration policy and remaining experiments

At runtime the components exchange structured state and evidence; training rewards are not added together. The Controller applies verification constraints after the Judge's forecast. This separation makes SFT, preference learning, verifiable outcomes and calibration independently measurable.

Before proposing joint training, pre-register comparisons: SFT-only; SFT+DPO; SFT+RLVR; SFT+DPO+RLVR; each with rule Judge versus separately calibrated Judge. Evaluate budget, abstention, domain shift, fabricated citations and verifier exploitation. Any future combined scalar objective or constrained optimization must state units, coefficients, constraint thresholds, failure semantics and reference distributions, and compare against those separated baselines.

## Symbolic stages

`task: symbolic` selects separately versioned binomial data for the SFT, preference and RLVR trainer. Symbolic and legacy checkpoints cannot be interchanged silently. The separate `symbolic-judge-train` entry point fits a symbolic checker-PASS forecast, with its own checkpoint kind, features, calibration split and tested exact resume. See [objectives, data and commands](SYMBOLIC.md).

Programmatic preferences do not constitute collected human feedback. Symbolic UNKNOWN is a zero reward for the precisely defined checker-PASS target, not a false theorem label. The [first build](../reports/phase-2c-implementation.md) saved all four models but achieved zero verified free-generation test answers; finite-candidate RLVR scores were not a reliable indicator of generation quality. No joint objective or model promotion follows.

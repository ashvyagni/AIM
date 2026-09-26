# Phase 2C — Bounded symbolic implementation

## Architecture and scope

This adds a second, explicitly versioned domain to Model-1. The numerical prediction Controller and its canonical suites remain separate. `aim-polynomial-identity-v1` means equality of formal univariate polynomials over rational coefficients, under the exact assumptions `{"ring":"Q[x]","variable":"x"}`. It is not verification of a scientific law, natural-language entailment, or a proof assistant.

The system retains the approved architecture:

1. `SymbolicResearcher` supplies a plan and typed hypotheses.
2. `SymbolicController` retrieves matching expression declarations from immutable local sources. It passes copies of state to model components.
3. `SymbolicJudge` returns VERIFY/ABSTAIN and an optional forecast for a **symbolic checker-PASS event** before dispatch. Old numerical Judge checkpoints are rejected.
4. `CHECK_POLYNOMIAL_IDENTITY` runs in the existing bounded subprocess worker, with a termination deadline.
5. `SymbolicVerifier` checks the returned certificate by recomputing the bounded exact normal form. Source-span integrity is checked separately.
6. The Controller revises claims and renders a scoped response with check IDs and original source links. A read-only audit replays the ledger and checks resolved identities again.

The reference Researcher uses the elementary formula `(x+a)^2 = x^2+2ax+a^2`, for explicitly formatted integer offsets. Its code is separate from the verifier's AST and rational-polynomial implementation. It intentionally declines other forms. It is an algorithmic baseline, not learned intelligence. The native transformer adapter generates structured JSON and fails closed on invalid output, without reference-backend substitution.

## Checker contract

Accepted expressions: integer literals, `x`, parentheses, unary signs, addition, subtraction, multiplication, division by a polynomial that normalizes to a nonzero constant, and literal integer powers 0 through 16. In the formal polynomial convention used here, any polynomial to power zero is 1, including `0**0`. Variable denominators, floats, other variables, calls, attributes, indexing, comparisons, Boolean values, comprehensions and computed exponents are unsupported. Source text is data and cannot issue tool commands.

Hard limits per request: 512 characters and 128 AST nodes per expression; degree at most 16; rational coefficient numerator/denominator at most 128 bits; 8,192 counted arithmetic operations across both sides. Bounds apply to intermediate expressions too. An equivalent expression can exceed limits even when its simplified result is small. Calls use a default five-second worker deadline and Controller budget of at most three actions.

The checker builds sparse maps `degree -> Fraction`. Addition merges coefficients; multiplication convolves them. Unique canonical coefficients establish equality in Q[x]:

`p = q  iff  for every degree k, coefficient(p,k) = coefficient(q,k)`.

Both canonical forms, assumptions/request hash, checker version, limits, operation count and outcome are retained. A mathematical FAIL means two valid bounded polynomials differ. UNKNOWN means unsupported grammar or exceeded bounds; it must never be read as a disproof. ERROR/TIMEOUT indicate execution failures. A ToolResult PASS means the worker completed, and is distinct from the mathematical outcome in its value.

The checker and certificate replay share a trusted implementation. Replaying protects against certificate substitution and artifact corruption; it does not remove common-mode software bugs. Independent integer-convolution tests, known identities and adversarial cases reduce that risk. A future second CAS/proof-kernel adapter requires a separate experiment and dependency decision.

## Native Researcher and training

The existing randomly initialized byte-token transformer is reused without pretrained weights or tokenizer imports. New checkpoints declare `aim-symbolic-researcher-v1`. The smoke configuration is 90,624 parameters, context 256, width 64, two layers. This is an engineering size, not an approved final Researcher size. The existing two-million-parameter allocation guard remains.

Procedural binomial offsets are partitioned by complete group: train -12..11, validation 12..19, engineering test 20..27. These are only 40 closely related problems and become exposed fixtures. They cannot establish general algebraic reasoning. E0 refers to the stored expression declaration; exact reference resolution does not imply that the source already contains the proposed answer.

Separate configurations preserve separate checkpoints and objectives:

- SFT: mean response-token negative log likelihood, including EOS.
- Preference: DPO against the frozen parent, with `-log sigmoid(beta * [(log pi(chosen)-log pi(rejected)) - (log ref(chosen)-log ref(rejected))])`. Preferences are **programmatic**, not collected human feedback or an RLHF effectiveness result. Existing human-annotation intake remains available separately.
- RLVR: a sampled finite-candidate REINFORCE update with detached expected-reward baseline and frozen-reference KL. The three explicit responses are a correct expansion, an incorrect constant term, and an unsupported division. Reward is 1 only when strict parsing, the E0 contract, and exact identity checking pass. Rewards do not read the chosen field. This is a bounded action distribution, not free-form online generation RL. UNKNOWN receives zero because the event is checker PASS, not because UNKNOWN is mathematically false.
- Judge: separate 225-parameter five-input MLP, log/Brier proper score. Inputs are normalized expression lengths and counts of `x`, `*`, `/`; no normal forms or checker outcomes are inference features. Train labels come from the actual checker. Temperature is selected on the disjoint validation offsets from {.5,1,2,4} by log loss. Test offsets are loaded only after fitting and temperature selection.

Judge checking policy: with illustrative success reward 1 and check cost c, VERIFY when p > c, else ABSTAIN. This is a one-step utility rule. It is not learned information acquisition or a complete sequential RLCD algorithm. The calibration target counts valid false identities and unsupported proposals as non-PASS. No numerical-domain calibration transfers implicitly. Train-prior forecasts are reported alongside Judge test scores.

Every learned component is opt-in. No combined reward, default promotion or new scale commitment follows from this build. All stages support exact CPU continuation through optimizer/RNG/configuration/dataset state. The symbolic Judge uses its own checkpoint kind and fitting-data hash. Test-data changes cannot alter its fitting-data identity.

## Commands

Run from `AIM/model-1/` in the pinned environment:

```sh
.venv/bin/python -m aim symbolic-loop
.venv/bin/python -m aim symbolic-audit runs/<symbolic-loop-run>
.venv/bin/python -m aim symbolic-evaluate
.venv/bin/python -m aim train --config configs/symbolic-sft.json
.venv/bin/python -m aim train --config configs/symbolic-preference.json --initialize runs/<sft-run>/checkpoint.pt
.venv/bin/python -m aim train --config configs/symbolic-rlvr.json --initialize runs/<preference-run>/checkpoint.pt
.venv/bin/python -m aim symbolic-judge-train
.venv/bin/python -m aim symbolic-loop --researcher-checkpoint runs/<rlvr-run>/checkpoint.pt --judge runs/<judge-run>/checkpoint.pt
.venv/bin/python -m aim.symbolic_reproduce --export reports/<new-evidence-directory>
```

`--case` supplies a strict JSON case to `symbolic-loop`; omitting it creates the documented binomial demo. `--max-actions 0` exercises unresolved budget behavior. `--check-cost` changes only the optional learned Judge's illustrative decision threshold. `--resume` works for the existing `train` command and `symbolic-judge-train`; the requested total steps must advance the checkpoint. Data/checkpoint/config changes are rejected on resume.

The reproduction runs all tests, trains SFT → preference → RLVR and the separate Judge, then checks the reference, three Researcher checkpoints, reference+Judge and RLVR+Judge. Each evaluation executes the 18-case canonical symbolic regression suite and eight actual research-loop episodes. All episodes receive read-only audits. Export directories are created exclusively; full run directories, weights, source snapshots and SQLite ledgers remain local.

## Build acceptance, recorded before the full smoke run

Required: all existing and new tests pass; canonical symbolic expected outcomes match; deterministic reference completes the eight supported expansions; resolved claims have exact checks and valid provenance; each native training stage saves a readable checkpoint; exact continuation tests pass; malformed neural output, abstention, unsupported syntax and tool failures remain unresolved. Existing numerical suite files must be unchanged.

Learned free-generation success is measured and reported, with no minimum quality threshold or promotion claim. This is a single-seed integration smoke run, not a causal comparison of algorithms. Preference/RLVR change training budgets and follow the preceding stage; there are no matched-budget control arms. No VIT physical cluster throughput, large-model feasibility, or scientific discovery is measured here.

## Next work

Add a richer independently generated algebra task distribution and a second implementation for differential verification before making capability claims. Add human feedback only with explicit annotation provenance. Calibration transfer to genuinely generated proposals needs fresh groups and resolved outcomes. The larger corpus, tokenizer and 100M–300M proxy build remains dependent on corpus preparation and actual hardware measurements.

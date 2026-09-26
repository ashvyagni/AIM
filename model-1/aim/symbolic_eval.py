"""Real symbolic tool dispatches and independently audited research-loop episodes."""
import json
from pathlib import Path

from .contracts import Action, ContractError, Outcome
from .symbolic import identity_request, check_identity
from .symbolic_data import symbolic_case, symbolic_data
from .symbolic_loop import SymbolicController, SymbolicTransformerResearcher, audit_symbolic_run
from .tools import ToolRunner
from .tracking import Run, file_hash, write_json


def evaluate_symbolic(suite, runs, checkpoint=None, judge=None):
    suite = Path(suite)
    expected_hash = suite.with_suffix(".sha256").read_text().strip()
    if file_hash(suite) != expected_hash:
        raise ContractError("symbolic canonical suite checksum mismatch")
    data = json.loads(suite.read_text())
    if data.get("schema") != "aim-symbolic-eval-v1":
        raise ContractError("invalid symbolic evaluation suite")
    with Run(Path(runs), "symbolic-evaluation", {"suite_hash": expected_hash,
             "researcher_checkpoint": str(checkpoint) if checkpoint else None,
             "judge_model_id": getattr(judge, "model_id", "symbolic-verification-first-v1")},
             [suite]+([Path(checkpoint)] if checkpoint else [])) as run:
        checks = []
        for case in data["cases"]:
            request = identity_request(case["lhs"], case["rhs"])
            result = ToolRunner().execute(Action(case["id"], "CHECK_POLYNOMIAL_IDENTITY", request))
            actual = result.value["outcome"] if result.outcome == Outcome.PASS else result.outcome.value
            integrity = result.outcome == Outcome.PASS and result.value == check_identity(request)
            checks.append({"case": case, "tool": result.__dict__, "passed": integrity and actual == case["expected"]})
        researcher = SymbolicTransformerResearcher(checkpoint) if checkpoint else None
        episodes = []
        for row in symbolic_data("test"):
            case = symbolic_case(row["lhs"], row["id"])
            with Run(run.path/"episodes", "symbolic-case", {"case": case,
                     "researcher_checkpoint": str(checkpoint) if checkpoint else None,
                     "judge_model_id": getattr(judge, "model_id", "symbolic-verification-first-v1")},
                     [Path(checkpoint)] if checkpoint else []) as episode:
                state = SymbolicController(researcher=researcher, judge=judge).run(case, episode)
            audit = audit_symbolic_run(episode.path)
            episodes.append({"group": row["group"], "path": str(episode.path.relative_to(run.path)),
                             "audit": audit, "state": state.to_dict()})
        metrics = {"checks_passed": sum(c["passed"] for c in checks), "checks_total": len(checks),
                   "episodes": len(episodes), "valid_proposals": sum(bool(e["state"]["hypotheses"]) for e in episodes),
                   "verified_episodes": sum(e["audit"]["verified"] > 0 for e in episodes),
                   "tool_dispatches": sum(e["audit"]["dispatches"] for e in episodes),
                   "scope": "eight disjoint-offset binomial engineering cases; no model promotion threshold"}
        write_json(run.path/"results.json", {"metrics": metrics, "checker_cases": checks, "episodes": episodes})
        if metrics["checks_passed"] != metrics["checks_total"]:
            raise ContractError("symbolic regression suite failed; artifacts retained")
    return run.path, metrics

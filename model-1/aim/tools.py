"""Allowlisted action dispatch with process termination on timeout."""
import json
import subprocess
import sys
import time

from .contracts import Action, ContractError, Outcome, ToolResult
from .tracking import ROOT, canonical


class ToolRunner:
    version = "bounded-worker-v1"
    allowed = {"CALCULATE", "MEASURE"}

    def execute(self, action: Action) -> ToolResult:
        if action.name not in self.allowed:
            raise ContractError(f"Action is not implemented or permitted: {action.name}")
        payload = canonical({"operation": action.name, "arguments": action.arguments})
        if len(payload) > 65536:
            raise ContractError("Action exceeds input limit")
        started = time.perf_counter()
        try:
            completed = subprocess.run([sys.executable, "-m", "aim.tool_worker"], cwd=ROOT,
                                       input=payload, capture_output=True, text=True,
                                       timeout=action.timeout_seconds)
            output = json.loads(completed.stdout)
            outcome = Outcome.PASS if completed.returncode == 0 else Outcome.ERROR
            if outcome == Outcome.PASS and output.get("value") is None:
                outcome = Outcome.UNKNOWN
            detail = output.get("error", output.get("reason", output.get("method", "")))
            return ToolResult(action.id, outcome, output.get("value"), detail,
                              time.perf_counter() - started, self.version)
        except subprocess.TimeoutExpired:
            return ToolResult(action.id, Outcome.TIMEOUT, None, "worker killed at deadline",
                              time.perf_counter() - started, self.version)
        except (OSError, ValueError) as exc:
            return ToolResult(action.id, Outcome.ERROR, None, str(exc), time.perf_counter() - started, self.version)

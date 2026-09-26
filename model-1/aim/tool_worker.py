"""Bounded operations only. This process does not accept or execute generated code."""
import json
import sys
from decimal import Decimal, localcontext

from .contracts import ContractError, finite_number


def execute(request):
    if not isinstance(request, dict) or set(request) != {"operation", "arguments"}:
        raise ContractError("Malformed tool request")
    op, args = request["operation"], request["arguments"]
    if op == "CHECK_POLYNOMIAL_IDENTITY":
        from .symbolic import check_identity, VERSION
        return {"value": check_identity(args), "method": VERSION}
    if op == "CALCULATE":
        if set(args) != {"coefficients", "x"} or not 1 <= len(args["coefficients"]) <= 3:
            raise ContractError("Bad polynomial contract")
        x = finite_number(args["x"], 1e4)
        coefficients = [finite_number(c, 1e6) for c in args["coefficients"]]
        with localcontext() as context:
            context.prec = 40
            y = Decimal(0)
            for c in reversed(coefficients):
                y = y * Decimal(str(x)) + Decimal(str(c))
        return {"value": finite_number(float(y)), "method": "decimal-polynomial-v1"}
    if op == "MEASURE":
        if set(args) != {"x", "observations", "available"} or type(args["available"]) is not bool:
            raise ContractError("Bad measurement contract")
        if not args["available"]:
            return {"value": None, "method": "synthetic-measurement-v1", "reason": "measurement unavailable"}
        target = finite_number(args["x"])
        values = [finite_number(y) for x, y in args["observations"] if finite_number(x) == target]
        if len(values) != 1:
            return {"value": None, "method": "synthetic-measurement-v1", "reason": "missing or conflicting measurement"}
        return {"value": values[0], "method": "synthetic-measurement-v1"}
    raise ContractError("Operation is not allowed")


def main():
    try:
        raw = sys.stdin.read(65537)
        if len(raw) > 65536:
            raise ContractError("Tool input too large")
        print(json.dumps(execute(json.loads(raw)), allow_nan=False))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        raise SystemExit(2)


if __name__ == "__main__":
    main()

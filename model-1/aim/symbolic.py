"""Exact, deliberately bounded univariate polynomial identity checking.

No eval, generated code, numerical sampling, external CAS or learned verdicts.
The trusted implementation is still ordinary tested software, not a proof assistant.
"""
from __future__ import annotations

import ast
from fractions import Fraction

from .contracts import ContractError
from .tracking import digest

SCHEMA = "aim-polynomial-identity-v1"
ASSUMPTIONS = {"ring": "Q[x]", "variable": "x"}
VERSION = "exact-rational-normal-form-v1"
LIMITS = {"characters": 512, "ast_nodes": 128, "degree": 16,
          "coefficient_bits": 128, "operations": 8192}
SCOPE = "Equality of formal polynomials in Q[x]; exact rational coefficients; no empirical or general theorem-proving claim"


class Unsupported(ContractError):
    """Outside the declared grammar or resource bounds, not a false identity."""


class Normalizer:
    def __init__(self):
        self.operations = 0

    def bounded(self, polynomial):
        self.operations += 1
        if self.operations > LIMITS["operations"]:
            raise Unsupported("symbolic operation budget exceeded")
        result = {k: v for k, v in polynomial.items() if v}
        if any(k > LIMITS["degree"] or max(abs(v.numerator).bit_length(), v.denominator.bit_length()) >
               LIMITS["coefficient_bits"] for k, v in result.items()):
            raise Unsupported("degree or rational coefficient budget exceeded")
        return result

    def add(self, left, right, sign=1):
        out = dict(left)
        for degree, value in right.items():
            out[degree] = out.get(degree, Fraction(0)) + sign * value
        return self.bounded(out)

    def multiply(self, left, right):
        out = {}
        for a, x in left.items():
            for b, y in right.items():
                self.operations += 1
                out[a+b] = out.get(a+b, Fraction(0)) + x*y
        return self.bounded(out)

    def parse(self, expression):
        if not isinstance(expression, str) or not 1 <= len(expression) <= LIMITS["characters"]:
            raise Unsupported("expression character budget exceeded")
        try:
            tree = ast.parse(expression, mode="eval")
        except (SyntaxError, ValueError, RecursionError) as exc:
            raise Unsupported("invalid expression syntax") from exc
        if sum(1 for _ in ast.walk(tree)) > LIMITS["ast_nodes"]:
            raise Unsupported("expression AST budget exceeded")
        return self.visit(tree.body)

    def visit(self, node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return self.bounded({0: Fraction(node.value)})
        if isinstance(node, ast.Name) and node.id == "x":
            return {1: Fraction(1)}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return self.bounded({k: (-v if isinstance(node.op, ast.USub) else v)
                                 for k, v in self.visit(node.operand).items()})
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                # Exponents must be literal integers, never evaluated expressions.
                if not isinstance(node.right, ast.Constant) or type(node.right.value) is not int or not 0 <= node.right.value <= LIMITS["degree"]:
                    raise Unsupported("exponent must be a literal integer in 0..16")
                base, result = self.visit(node.left), {0: Fraction(1)}
                for _ in range(node.right.value):
                    result = self.multiply(result, base)
                return result
            if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                left, right = self.visit(node.left), self.visit(node.right)
                if isinstance(node.op, (ast.Add, ast.Sub)):
                    return self.add(left, right, -1 if isinstance(node.op, ast.Sub) else 1)
                if isinstance(node.op, ast.Mult):
                    return self.multiply(left, right)
                if set(right) != {0}:
                    raise Unsupported("division requires a nonzero constant polynomial")
                return self.bounded({k: v/right[0] for k, v in left.items()})
        raise Unsupported("unsupported syntax; only x, integers, +, -, *, constant / and bounded ** are permitted")


def identity_request(lhs, rhs):
    return {"schema": SCHEMA, "assumptions": dict(ASSUMPTIONS), "lhs": lhs, "rhs": rhs}


def check_identity(request):
    if not isinstance(request, dict) or set(request) != {"schema", "assumptions", "lhs", "rhs"}:
        raise ContractError("malformed symbolic identity request")
    if request["schema"] != SCHEMA or request["assumptions"] != ASSUMPTIONS:
        raise ContractError("unsupported symbolic schema or assumptions")
    normalizer = Normalizer()
    result = {"schema": SCHEMA, "checker": VERSION, "request_hash": digest(request),
              "scope": SCOPE, "limits": dict(LIMITS)}
    try:
        left, right = normalizer.parse(request["lhs"]), normalizer.parse(request["rhs"])
        encode = lambda p: [[k, str(v.numerator), str(v.denominator)] for k, v in sorted(p.items())]
        result.update(outcome="PASS" if left == right else "FAIL", lhs_normal=encode(left),
                      rhs_normal=encode(right), detail="exact normal forms equal" if left == right else "exact normal forms differ")
    except Unsupported as exc:
        result.update(outcome="UNKNOWN", lhs_normal=None, rhs_normal=None, detail=str(exc))
    result["operations"] = normalizer.operations
    return result

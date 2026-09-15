"""Slice 8 — arithmetic AST evaluator.

`python_calc` is the one new tool in Slice 8. The brief forbids raw `eval()`
on strings: this module parses the expression into an `ast`, walks it, and
rejects everything except literal numbers, parentheses, and the binary /
unary operators `+ - * / // % **`. Names, attribute access, calls,
subscripts, comprehensions, and any other AST node are policy errors — the
worker module converts them to a structured `{error: ...}` response.

The evaluator is pure (no imports, no I/O) and runs both in-process (when
a test pins `backend: inprocess` for `python_calc`) and in the subprocess
worker. Either way, the same code path is exercised, so a sandboxed run
and an in-process test get identical answers.
"""
from __future__ import annotations

import ast
import operator
from typing import Any


# Map AST operator nodes to Python callables. Restricted to numeric
# arithmetic; bitwise ops are out (they're rarely useful for tool calls
# and easy to misuse).
_BIN_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalcError(Exception):
    """Raised on any policy or evaluation failure. Worker catches and reports."""


def evaluate(expression: str) -> float | int:
    """Evaluate `expression` against the AST whitelist.

    Returns the result as int when it has no fractional part and fits in a
    Python int, otherwise as float. A round-trip through float keeps the
    contract simple — `2*(3+4)` returns `int(14)`, `1/3` returns
    `float(0.3333…)`.

    Raises `CalcError` with a short, structured message; the worker
    surfaces that message as `{error: ...}` on stdout.
    """
    if not isinstance(expression, str):
        raise CalcError("expression must be a string")
    text = expression.strip()
    if not text:
        raise CalcError("empty expression")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise CalcError(f"syntax error: {exc.msg}") from exc

    result = _walk(tree.body)
    # Coerce int-looking floats so callers get a stable JSON number.
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result


def _walk(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _walk(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise CalcError(f"unsupported constant: {type(node.value).__name__}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise CalcError(f"unsupported operator: {op_type.__name__}")
        left = _walk(node.left)
        right = _walk(node.right)
        try:
            return _BIN_OPS[op_type](left, right)
        except ZeroDivisionError as exc:
            raise CalcError("division by zero") from exc
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise CalcError(f"unsupported unary operator: {op_type.__name__}")
        return _UNARY_OPS[op_type](_walk(node.operand))
    # Anything else is a policy reject — names, attribute access, calls,
    # subscripts, comprehensions, comparisons, etc.
    raise CalcError(f"disallowed syntax: {type(node).__name__}")


# ── Worker entry point ────────────────────────────────────────────────────


def main() -> int:
    """Worker entry point for the subprocess backend.

    Reads `expression` from `--expression` (CLI) or stdin (JSON), writes
    `{"result": ...}` or `{"error": ...}` to stdout. Exit code 0 on a
    successful evaluation (including policy-rejected — those are reported
    as `{"error": ...}`, not as non-zero exit, so the executor sees a
    structured response).
    """
    import json
    import sys

    expression = ""
    argv = sys.argv[1:]
    if "--expression" in argv:
        i = argv.index("--expression")
        if i + 1 < len(argv):
            expression = argv[i + 1]
    if not expression and not sys.stdin.isatty():
        try:
            stdin_payload = sys.stdin.read().strip()
            if stdin_payload:
                payload = json.loads(stdin_payload)
                expression = str(payload.get("expression", ""))
        except Exception:  # noqa: BLE001 — stdin is best-effort
            pass

    try:
        result = evaluate(expression)
        sys.stdout.write(json.dumps({"ok": True, "result": result}))
    except CalcError as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": str(exc)}))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Slice 8 — subprocess worker entry point.

The subprocess backend runs `python <worker.py> --tool <name> ...`. The
worker is intentionally minimal: dispatch one tool name, run a tiny
handler, print a JSON result, and exit. **Heavy deps are not loaded in
this process** — the worker does not import the `contextlab` package,
so sentence-transformers / torch / numpy never start. That keeps the
cold-start cost low (well under the wall-clock timeout) and means a
runaway worker can't trigger OpenBLAS-side allocations.

Tools supported by this worker:

  - python_calc     Arithmetic AST evaluator. Production tool.
  - sleep_forever   Test fixture. Sleeps `--seconds` past the worker's
                    timeout, exercising the subprocess.TimeoutExpired
                    path in the parent. Not in the production tool
                    registry (only added by tests).
  - echo_env        Test fixture. Writes the worker's `os.environ` to
                    stdout as JSON. Used to verify parent secrets do
                    not leak. Not in the production tool registry.
  - read_secret     Test fixture. Reads `--file_path` and writes its
                    contents to stdout. The path-jail test asserts the
                    parent denies this *before* the worker is launched,
                    so the worker never sees a non-allowlisted path.
                    Not in the production tool registry.

The production-only `python_calc` evaluator is duplicated from
`contextlab.sandbox.calc` on purpose: this file is the *only* code the
worker runs. If you change the calc policy, change both — the
duplication is a hard ban on accidentally letting the worker import
the rest of the project.

Network is off by default: the parent passes no proxy env vars and the
worker has no business making outbound requests. The test for env
non-leak sets `OPENAI_API_KEY` in the parent and asserts the worker
never sees it.
"""
from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import sys
import time
from typing import Callable


# ── python_calc evaluator (duplicated from contextlab.sandbox.calc) ─────────
# Names, attribute access, calls, subscripts, comprehensions are all
# rejected. Only literal numbers, parentheses, and the binary / unary
# operators below are allowed.

_BIN_OPS: dict = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class _CalcError(Exception):
    """Policy / evaluation failure. main() converts to JSON."""


def _walk(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _walk(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise _CalcError(f"unsupported constant: {type(node.value).__name__}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise _CalcError(f"unsupported operator: {op_type.__name__}")
        left = _walk(node.left)
        right = _walk(node.right)
        try:
            return _BIN_OPS[op_type](left, right)
        except ZeroDivisionError as exc:
            raise _CalcError("division by zero") from exc
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise _CalcError(f"unsupported unary operator: {op_type.__name__}")
        return _UNARY_OPS[op_type](_walk(node.operand))
    raise _CalcError(f"disallowed syntax: {type(node).__name__}")


def _evaluate(expression: str):
    if not isinstance(expression, str):
        raise _CalcError("expression must be a string")
    text = expression.strip()
    if not text:
        raise _CalcError("empty expression")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise _CalcError(f"syntax error: {exc.msg}") from exc
    result = _walk(tree.body)
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result


# ── Test-fixture handlers ──────────────────────────────────────────────────


def _handle_sleep_forever(args: dict) -> dict:
    seconds = float(args.get("seconds", "60") or 60)
    time.sleep(seconds)
    return {"slept": seconds}


def _handle_echo_env(args: dict) -> dict:
    marker = args.get("marker", "")
    return {
        "marker": marker,
        "env": dict(os.environ),
    }


def _handle_read_secret(args: dict) -> dict:
    path = args.get("file_path", "") or ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "path": path}
    return {"path": path, "content": content}


# ── Dispatch ────────────────────────────────────────────────────────────────


_HANDLERS: dict[str, Callable[[dict], dict]] = {
    "python_calc": lambda a: {"result": _evaluate(a.get("expression", ""))},
    "sleep_forever": _handle_sleep_forever,
    "echo_env": _handle_echo_env,
    "read_secret": _handle_read_secret,
}


def main(argv: list[str] | None = None) -> int:
    """Read --tool and --args-json, dispatch to a handler, write JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True)
    parser.add_argument("--args-json", default="{}")
    raw = argv if argv is not None else sys.argv[1:]
    try:
        args = parser.parse_args(raw)
    except SystemExit:
        sys.stdout.write(json.dumps({"ok": False, "error": "bad argv", "code": "policy"}))
        sys.stdout.flush()
        return 0

    try:
        kwargs = json.loads(args.args_json) if args.args_json else {}
    except json.JSONDecodeError:
        kwargs = {}

    handler = _HANDLERS.get(args.tool)
    if handler is None:
        sys.stdout.write(
            json.dumps({"ok": False, "error": f"unknown tool: {args.tool}", "code": "policy"})
        )
        sys.stdout.flush()
        return 0

    try:
        result = handler(kwargs)
        sys.stdout.write(json.dumps({"ok": True, "result": result}))
    except _CalcError as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": str(exc), "code": "policy"}))
    except Exception as exc:  # noqa: BLE001 — runtime errors are reported, not raised
        sys.stdout.write(
            json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}", "code": "runtime"})
        )
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
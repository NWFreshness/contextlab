"""Slice 8 — sandboxed tool executor.

Replaces the Slice 7 InProcessExecutor as the default executor when
`config/sandbox.yaml: backend == "subprocess"`. The brief keeps the Executor
protocol stable so the orchestrator's wiring is unchanged.

Public API:
    SandboxLimits / ToolSandboxPolicy   — config models.
    SandboxPolicy / ToolError / ErrorCode — the policy gate.
    SandboxedExecutor                   — the dispatching executor.
    load_sandbox_config                 — config loader.
    evaluate(expression)                — the AST calc evaluator.
"""

from contextlab.sandbox.calc import CalcError, evaluate
from contextlab.sandbox.executor import SandboxedExecutor
from contextlab.sandbox.limits import (
    Backend,
    SandboxLimits,
    ToolSandboxPolicy,
    has_prlimit,
    load_sandbox_config,
)
from contextlab.sandbox.policy import ErrorCode, SandboxPolicy, ToolError

__all__ = [
    "Backend",
    "CalcError",
    "ErrorCode",
    "SandboxLimits",
    "SandboxPolicy",
    "SandboxedExecutor",
    "ToolError",
    "ToolSandboxPolicy",
    "evaluate",
    "has_prlimit",
    "load_sandbox_config",
]
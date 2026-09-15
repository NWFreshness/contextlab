"""Slice 8 — sandbox policy gate.

Every tool call passes through `SandboxPolicy.check` before the executor
runs anything. Three things fail closed:

    1. Unknown tool            -> ToolError(code="unknown_tool")
    2. Args outside the schema -> ToolError(code="policy")
    3. Path args outside the
       configured allowlist    -> ToolError(code="policy")

The policy is pure: it returns either an `Ok` (decision to proceed) or a
`ToolError`. The executor catches `ToolError`, attaches the code to the
`tool.exec` span, and returns a `ToolResult` whose content is the
structured error payload — so the trajectory shows what was denied.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from contextlab.sandbox.limits import SandboxLimits, ToolSandboxPolicy


# Structured error codes. The brief pins these names; do not rename
# without updating trace consumers.
class ErrorCode:
    UNKNOWN_TOOL = "unknown_tool"
    TIMEOUT = "timeout"
    POLICY = "policy"
    SANDBOX = "sandbox"
    RUNTIME = "runtime"


class ToolError(Exception):
    """Structured tool failure. The executor turns this into a ToolResult.

    `code` is one of the strings in `ErrorCode`. `message` is short and
    free of host paths so it cannot leak the workdir tree to the
    trajectory.
    """

    def __init__(self, code: str, message: str, *, tool: Optional[str] = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.tool = tool

    def as_dict(self) -> dict[str, str]:
        return {"error": self.message, "code": self.code, "tool": self.tool or ""}


def _normalize(path: Path) -> Path:
    """Resolve `..` and symlinks. Comparison base for the path jail."""
    return path.resolve()


class SandboxPolicy:
    """The policy layer. Always-on, fail-closed."""

    def __init__(self, limits: SandboxLimits, *, project_root: Optional[Path] = None) -> None:
        self.limits = limits
        self.project_root = (
            Path(project_root) if project_root else Path(__file__).resolve().parents[3]
        )
        # Pre-resolve the allowlist once. `path_allowlist` is host-relative
        # so we anchor against the project root; tests can override the
        # root via the constructor.
        self._allowed_paths: list[Path] = [
            _normalize(self.project_root / p) for p in limits.path_allowlist
        ]
        self._env_allow: set[str] = set(limits.env_allowlist)

    # ── Tool surface ────────────────────────────────────────────────────

    def registered_tools(self) -> list[str]:
        """Tool names the policy knows about. Includes the default
        `sandbox: true` fallback for any name not explicitly listed."""
        explicit = set(self.limits.tools.keys())
        return sorted(explicit)

    def tool_policy(self, tool: str) -> ToolSandboxPolicy:
        """Effective per-tool policy. Missing -> default `sandbox: true`."""
        return self.limits.tools.get(tool, ToolSandboxPolicy(sandbox=True))

    # ── Gates ───────────────────────────────────────────────────────────

    def check(self, tool: str, args: dict[str, Any]) -> None:
        """Raise `ToolError` on any policy failure. Returns None on ok."""
        if tool not in self.limits.tools:
            raise ToolError(
                ErrorCode.UNKNOWN_TOOL,
                f"tool {tool!r} is not registered",
                tool=tool,
            )
        self._check_args(tool, args)
        self._check_paths(tool, args)

    def _check_args(self, tool: str, args: dict[str, Any]) -> None:
        schema = set(self.tool_policy(tool).args_schema)
        unknown = set(args.keys()) - schema
        if unknown:
            raise ToolError(
                ErrorCode.POLICY,
                f"tool {tool!r} does not accept arguments: {sorted(unknown)}",
                tool=tool,
            )

    def _check_paths(self, tool: str, args: dict[str, Any]) -> None:
        """Reject any `path` / `*_path` / `file` arg that resolves outside
        the project root + `path_allowlist`."""
        for key, value in args.items():
            if not _looks_like_path_arg(key):
                continue
            if not isinstance(value, str):
                raise ToolError(
                    ErrorCode.POLICY,
                    f"tool {tool!r}: arg {key!r} must be a string path",
                    tool=tool,
                )
            try:
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = (self.project_root / candidate)
                resolved = _normalize(candidate)
            except OSError as exc:
                raise ToolError(
                    ErrorCode.POLICY,
                    f"tool {tool!r}: cannot resolve {key!r}: {exc.strerror or exc}",
                    tool=tool,
                ) from exc
            # The workdir is always accessible — checked separately by the
            # executor (one workdir per run). Anything outside the
            # allowlist is denied here.
            if not self._is_allowed(resolved):
                raise ToolError(
                    ErrorCode.POLICY,
                    f"tool {tool!r}: arg {key!r} is not in the path allowlist",
                    tool=tool,
                )

    def _is_allowed(self, resolved: Path) -> bool:
        for allowed in self._allowed_paths:
            try:
                resolved.relative_to(allowed)
                return True
            except ValueError:
                continue
        return False

    # ── Env gate ────────────────────────────────────────────────────────

    def env_for_child(self, parent_env: dict[str, str]) -> dict[str, str]:
        """Build the env passed to the subprocess.

        Anything not in `env_allowlist` is dropped. The brief requires
        parent secrets (OPENAI_API_KEY and friends) not to leak — by
        construction, since they are not in the allowlist.
        """
        return {k: v for k, v in parent_env.items() if k in self._env_allow}


def _looks_like_path_arg(name: str) -> bool:
    """Heuristic for path-typed arguments. Keeps the policy narrow so a
    `chunk_id` arg (e.g. `incident_runbook::c0004`) doesn't get treated as
    a path and accidentally rejected."""
    lowered = name.lower()
    return lowered in {"path", "file", "filepath", "filename"} or lowered.endswith(
        "_path"
    ) or lowered.endswith("_file")
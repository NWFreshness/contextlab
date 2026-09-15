"""Slice 8 — sandboxed executor.

`SandboxedExecutor` implements the Slice 7 `Executor` protocol
(`run(tool: str, args: dict) -> ToolResult`). It routes every call
through `SandboxPolicy.check` and then either:

    - `sandbox: false` tools -> the Slice 7 `InProcessExecutor` (so
      retrieve / read_chunk stay fast and in-process; the blast radius
      is the local indexes, which is the whole point of Slice 1).
    - `sandbox: true`  tools -> `subprocess.run` with a fresh workdir,
      env allowlist, timeout, and an output cap. `prlimit` is used for
      CPU/memory caps when available; missing on PATH just means no
      caps (best-effort).

Every call emits a `tool.exec` span with `tool`, `sandbox`, `backend`,
`timeout_s`, `exit_code`, and `code` on error. Unknown tools raise
`UnknownToolError` so the orchestrator's stop_reason wiring is unchanged
from Slice 7.

The executor never raises on policy / timeout / runtime errors — those
become structured `ToolResult`s with `{"error": ..., "code": ...}` so
the trajectory shows what went wrong and Slice 9 can grade it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from contextlab.orch.executor_inprocess import (
    Executor,
    InProcessExecutor,
    UnknownToolError,
)
from contextlab.sandbox.calc import CalcError, evaluate
from contextlab.sandbox.limits import SandboxLimits, ToolSandboxPolicy, has_prlimit, load_sandbox_config
from contextlab.sandbox.policy import ErrorCode, SandboxPolicy, ToolError
from contextlab.trace import start_span
from contextlab.types import ToolResult


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class SandboxedExecutor:
    """Subprocess-backed executor with a policy gate in front.

    Constructed by the orchestrator. Tests can pass a custom `limits` to
    point the workdir at `tmp_path / "sandbox_work"` and shrink the
    timeout.
    """

    def __init__(
        self,
        limits: Optional[SandboxLimits] = None,
        *,
        project_root: Optional[Path] = None,
        in_process_fallback: Optional[Executor] = None,
    ) -> None:
        self.limits = limits or load_sandbox_config()
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self.policy = SandboxPolicy(self.limits, project_root=self.project_root)
        # In-process helpers used when `sandbox: false`. Slice 1's
        # retrieve / read_chunk go here.
        self._inproc = in_process_fallback or InProcessExecutor(
            chunks_path=self.project_root / "data" / "chunks.jsonl",
            registered_tools=sorted(self.limits.tools.keys()),
        )
        # Detect optional best-effort backends. `prlimit` is on most Linux
        # systems; missing just means no CPU/memory caps.
        self._has_prlimit = has_prlimit()
        # The worker (`worker.py`) handles `python_calc` plus any
        # test-only fixtures (sleep_forever, echo_env, read_secret) that
        # happen to appear in `limits.tools` with `sandbox: true`.
        # Production code only ships `python_calc`; tests extend
        # `limits.tools` to add fixtures.
        self._sandboxed_tools: set[str] = {
            name
            for name, pol in self.limits.tools.items()
            if pol.sandbox
        }

    # ── Public surface ───────────────────────────────────────────────────

    def run(self, tool: str, args: dict[str, Any]) -> ToolResult:
        """Dispatch one tool call. Always returns a ToolResult.

        Unknown tools propagate `UnknownToolError` so the orchestrator
        sets `stop_reason=unknown_tool`. All other failures are encoded
        in the result so the trajectory stays inspectable.
        """
        tool_policy = self.policy.tool_policy(tool)
        sandbox = tool_policy.sandbox

        # The policy gate runs *before* the span so a denial can never be
        # confused with a real execution — `code="unknown_tool"` on the
        # span is the marker.
        with start_span(
            "tool.exec",
            tool=tool,
            sandbox=sandbox,
            backend=self.limits.backend if sandbox else "inprocess",
            timeout_s=self.limits.timeout_s if sandbox else 0.0,
            args_keys=sorted((args or {}).keys()),
        ) as span:
            try:
                self.policy.check(tool, args or {})
            except ToolError as exc:
                span.set_attributes(code=exc.code, exit_code=-1)
                # An unknown tool bubbles up as `UnknownToolError` so the
                # orchestrator's Slice 7 catch fires and the run stops
                # with `stop_reason=unknown_tool`. Other policy errors
                # (extra args, path-jail violations) return a structured
                # `ToolResult` so the trajectory shows what was denied.
                if exc.code == ErrorCode.UNKNOWN_TOOL:
                    raise UnknownToolError(tool) from exc
                return _error_result(tool, exc)

            if not sandbox:
                span.set_attributes(backend="inprocess", code="ok", exit_code=0)
                try:
                    return self._inproc.run(tool, args or {})
                except UnknownToolError:
                    raise
                except Exception as exc:  # noqa: BLE001 — in-process tool failure is a ToolResult
                    span.fail(exc)
                    return _runtime_error_result(tool, exc)

            # Sandbox: subprocess path.
            return self._run_subprocess(tool, args or {}, span)

    # ── Subprocess backend ──────────────────────────────────────────────

    def _run_subprocess(
        self,
        tool: str,
        args: dict[str, Any],
        span: Any,
    ) -> ToolResult:
        if tool not in self._sandboxed_tools:
            # The only sandboxed tool Slice 8 ships is `python_calc`.
            # Test-only handler fixtures may add others via
            # `extra_sandbox_handlers=...`.
            span.set_attributes(code=ErrorCode.POLICY, exit_code=-1)
            return _error_result(
                tool,
                ToolError(
                    ErrorCode.POLICY,
                    f"sandboxed tool {tool!r} is not implemented",
                    tool=tool,
                ),
            )

        # Fresh workdir under `workdir_parent`. The brief pins a fresh
        # temp directory per run so a previous run's leftovers can't
        # influence this one.
        workdir_parent = Path(self.limits.workdir_parent)
        workdir_parent.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix="ctx-", dir=str(workdir_parent)))

        try:
            child_env = self.policy.env_for_child(dict(os.environ))
            # Invoke the worker as a script by absolute path. `python -m`
            # would import the `contextlab` package and pull in
            # sentence-transformers + torch + OpenBLAS, blowing the
            # wall-clock timeout before the worker even runs. By passing
            # the file directly, only `worker.py` is loaded — the AST
            # eval needs stdlib only.
            worker_path = Path(__file__).resolve().parent / "worker.py"
            argv = [
                sys.executable,
                str(worker_path),
                "--tool",
                tool,
                "--args-json",
                json.dumps(args or {}),
            ]

            # Optional: cap CPU via prlimit when on PATH AND `cpu_s` is set.
            #
            # `prlimit --cpu=N` is in CPU-seconds. Python startup of the
            # `contextlab` package pulls in sentence-transformers + torch
            # + OpenBLAS, which alone can exceed 2 CPU-seconds on cold
            # import. So `prlimit` is opt-in: the brief lists `cpu_s: 2`
            # as a *target*, but the wall-clock timeout (`timeout_s`) is
            # the verified containment. If `cpu_s` is None we skip
            # prlimit entirely (and log `backend: subprocess` only).
            if self._has_prlimit and self.limits.cpu_s is not None and self.limits.cpu_s > 0:
                argv = [
                    "prlimit",
                    f"--cpu={int(self.limits.cpu_s)}",
                    "--",
                    *argv,
                ]

            try:
                completed = subprocess.run(
                    argv,
                    cwd=str(workdir),
                    env=child_env,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=self.limits.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                span.set_attributes(
                    code=ErrorCode.TIMEOUT,
                    exit_code=-1,
                    timeout_s=self.limits.timeout_s,
                )
                return _error_result(
                    tool,
                    ToolError(
                        ErrorCode.TIMEOUT,
                        f"timed out after {self.limits.timeout_s}s",
                        tool=tool,
                    ),
                )

            stdout = (completed.stdout or "")[: self.limits.max_output_bytes]
            stderr = (completed.stderr or "")[: self.limits.max_output_bytes]

            span.set_attributes(exit_code=completed.returncode)

            if completed.returncode != 0:
                # The worker writes structured errors to stdout even on
                # policy rejects, so a non-zero exit is unexpected. Treat
                # it as runtime.
                span.set_attributes(code=ErrorCode.RUNTIME)
                return _runtime_error_result(
                    tool,
                    RuntimeError(
                        f"exit {completed.returncode}; stderr={stderr.strip()[:200]}"
                    ),
                )

            try:
                payload = json.loads(stdout) if stdout.strip() else {}
            except json.JSONDecodeError:
                span.set_attributes(code=ErrorCode.RUNTIME)
                return _runtime_error_result(
                    tool,
                    RuntimeError(f"worker stdout is not JSON: {stdout[:120]!r}"),
                )

            if payload.get("ok") is True:
                span.set_attributes(code="ok")
                # The worker's handlers wrap the result in `{"result": ...}`
                # so a future handler can add other fields. Unwrap here so
                # the orchestrator and assembler see the bare payload.
                inner = payload.get("result")
                if isinstance(inner, dict) and set(inner.keys()) == {"result"}:
                    content_value = inner.get("result")
                else:
                    content_value = inner
                return ToolResult(
                    tool_id=f"{tool}:ok",
                    name=tool,
                    content=json.dumps({"result": content_value}),
                    must_keep=False,
                )

            # Worker reported a structured error (e.g. policy reject).
            code = str(payload.get("code") or ErrorCode.POLICY)
            message = str(payload.get("error") or "policy violation")
            span.set_attributes(code=code)
            return _error_result(tool, ToolError(code, message, tool=tool))
        finally:
            # Clean up the per-run workdir. `ignore_errors=True` so a
            # permission hiccup never leaves a child process alive.
            shutil.rmtree(workdir, ignore_errors=True)

    # ── Convenience: re-exported in-process executor ────────────────────

    @property
    def registered_tools(self) -> list[str]:
        return sorted(self.limits.tools.keys())


# ── Helpers ──────────────────────────────────────────────────────────────


def _error_result(tool: str, exc: ToolError) -> ToolResult:
    payload = exc.as_dict()
    return ToolResult(
        tool_id=f"{tool}:error",
        name=tool,
        content=json.dumps(payload),
        must_keep=False,
    )


def _runtime_error_result(tool: str, exc: BaseException) -> ToolResult:
    payload = {
        "error": f"{type(exc).__name__}: {exc}",
        "code": ErrorCode.RUNTIME,
        "tool": tool,
    }
    return ToolResult(
        tool_id=f"{tool}:runtime",
        name=tool,
        content=json.dumps(payload),
        must_keep=False,
    )


__all__ = [
    "Executor",
    "SandboxedExecutor",
    "UnknownToolError",
]
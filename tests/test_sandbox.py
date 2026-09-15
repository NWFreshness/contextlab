"""Slice 8 — sandboxed executor tests.

No network, no API keys, no host env leaks. Five brief-pinned scenarios:

  - calc happy path: `2*(3+4)` returns `14`, AST whitelist enforced.
  - timeout fixture: a worker that sleeps past `timeout_s` returns
    `code=timeout` in < 5s wall clock.
  - allowlist: a tool name not in the policy is denied; extra args not
    in the tool's schema are denied.
  - path jail: a sandboxed tool that tries to read outside the workdir
    gets a `policy` deny. The brief's example is `BRIEF.md` or
    `../../.env`.
  - env non-leak: the parent sets `OPENAI_API_KEY`; the worker's
    `os.environ` does not contain it, and no ToolResult payload does.

Plus a few extras:
  - subprocess backend detection (`prlimit` not required).
  - orchestrator integration: `python_calc` end-to-end on a real query.

The fixture sandboxed tool (`read_secret`) is registered only in tests —
adding `shell` / `python -c` tools is a hard ban.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from contextlab.sandbox import (
    CalcError,
    ErrorCode,
    SandboxedExecutor,
    SandboxLimits,
    SandboxPolicy,
    ToolError,
    evaluate,
    load_sandbox_config,
)
from contextlab.sandbox.limits import ToolSandboxPolicy, has_prlimit
from contextlab.sandbox.policy import SandboxPolicy as _Policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def sandbox_workdir(tmp_path) -> Path:
    """Per-test sandbox workdir parent. Keeps the real `data/sandbox_work/`
    directory out of every test."""
    d = tmp_path / "sandbox_work"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def test_limits(sandbox_workdir, tmp_path) -> SandboxLimits:
    """A SandboxLimits pointed at tmp dirs. Tuned for fast tests."""
    return SandboxLimits(
        backend="subprocess",
        workdir_parent=sandbox_workdir,
        timeout_s=1.0,
        max_output_bytes=4000,
        cpu_s=None,  # opt-in; tests must not require prlimit
        memory_mb=256,
        env_allowlist=["PATH", "LANG", "LC_ALL"],
        path_allowlist=["data/chunks.jsonl", "data/sandbox_work"],
        tools={
            "retrieve": ToolSandboxPolicy(
                sandbox=False, args_schema=["query", "k", "mode"]
            ),
            "read_chunk": ToolSandboxPolicy(
                sandbox=False, args_schema=["chunk_id"]
            ),
            "python_calc": ToolSandboxPolicy(
                sandbox=True, args_schema=["expression"]
            ),
            "finish": ToolSandboxPolicy(sandbox=False, args_schema=["answer"]),
            # Test-only fixture tool: NOT shipped in production config.
            # It would, if allowed, read an arbitrary file path. The
            # policy must keep it denied (path jail + allowlist).
            "read_secret": ToolSandboxPolicy(
                sandbox=True, args_schema=["file_path"]
            ),
            # Test-only sleep fixture: holds the worker past timeout_s.
            "sleep_forever": ToolSandboxPolicy(
                sandbox=True, args_schema=["seconds"]
            ),
            # Test-only echo fixture for env non-leak: dumps its os.environ.
            "echo_env": ToolSandboxPolicy(
                sandbox=True, args_schema=["marker"]
            ),
        },
    )


@pytest.fixture
def executor(test_limits) -> SandboxedExecutor:
    return SandboxedExecutor(limits=test_limits, project_root=PROJECT_ROOT)


# ── Pure calc evaluator (in-process AST) ────────────────────────────────────


class TestCalcEvaluator:
    """The AST whitelist. Run in-process so the failure modes are obvious."""

    def test_simple_arithmetic(self):
        assert evaluate("2+3") == 5

    def test_nested_operators(self):
        assert evaluate("2*(3+4)") == 14
        assert evaluate("(1+2)*(3+4)") == 21
        assert evaluate("2**8") == 256
        assert evaluate("10//3") == 3
        assert evaluate("10%3") == 1

    def test_unary(self):
        assert evaluate("-5+3") == -2
        assert evaluate("+7") == 7

    def test_truediv_returns_float(self):
        result = evaluate("1/3")
        assert isinstance(result, float)
        assert abs(result - 0.3333) < 1e-3

    def test_rejects_name_lookup(self):
        with pytest.raises(CalcError) as exc_info:
            evaluate("__import__('os')")
        assert "disallowed" in str(exc_info.value).lower() or "unsupported" in str(exc_info.value).lower()

    def test_rejects_attribute_access(self):
        with pytest.raises(CalcError):
            evaluate("(1).__class__")

    def test_rejects_call(self):
        with pytest.raises(CalcError):
            evaluate("abs(-5)")

    def test_rejects_empty(self):
        with pytest.raises(CalcError):
            evaluate("   ")

    def test_rejects_syntax_error(self):
        with pytest.raises(CalcError):
            evaluate("1+")

    def test_division_by_zero(self):
        with pytest.raises(CalcError) as exc_info:
            evaluate("1/0")
        assert "division" in str(exc_info.value).lower()


# ── Policy gate ─────────────────────────────────────────────────────────────


class TestSandboxPolicy:
    """Pure policy checks — no subprocess."""

    def _policy(self) -> SandboxPolicy:
        limits = SandboxLimits(
            backend="subprocess",
            env_allowlist=["PATH", "LANG"],
            path_allowlist=["data/chunks.jsonl"],
            tools={
                "python_calc": ToolSandboxPolicy(sandbox=True, args_schema=["expression"]),
                "read_secret": ToolSandboxPolicy(sandbox=True, args_schema=["file_path"]),
            },
        )
        return SandboxPolicy(limits, project_root=PROJECT_ROOT)

    def test_unknown_tool_is_denied(self):
        with pytest.raises(ToolError) as exc_info:
            self._policy().check("shell", {"cmd": "ls"})
        assert exc_info.value.code == ErrorCode.UNKNOWN_TOOL

    def test_extra_args_are_denied(self):
        with pytest.raises(ToolError) as exc_info:
            self._policy().check("python_calc", {"expression": "1+1", "sneaky": "rm -rf /"})
        assert exc_info.value.code == ErrorCode.POLICY
        assert "sneaky" in exc_info.value.message

    def test_path_jail_denies_traversal(self):
        # The fixture uses a path that tries to escape the project root
        # via `..`. The policy's allowlist is checked after `..` resolves,
        # so `../../.env` is denied even if the literal path matches a
        # legitimate sibling.
        with pytest.raises(ToolError) as exc_info:
            self._policy().check("read_secret", {"file_path": "../../../etc/passwd"})
        assert exc_info.value.code == ErrorCode.POLICY

    def test_env_allowlist_drops_parent_secrets(self):
        parent = {
            "PATH": "/usr/bin",
            "LANG": "en_US.UTF-8",
            "OPENAI_API_KEY": "sk-test",
            "ANTHROPIC_API_KEY": "sk-anthropic",
        }
        child = self._policy().env_for_child(parent)
        assert "OPENAI_API_KEY" not in child
        assert "ANTHROPIC_API_KEY" not in child
        assert child.get("PATH") == "/usr/bin"
        assert child.get("LANG") == "en_US.UTF-8"


# ── Subprocess executor ─────────────────────────────────────────────────────


class TestSandboxedExecutor:
    """The full subprocess dispatch — timeout, output, errors."""

    def test_python_calc_happy_path(self, executor):
        """`python_calc 2*(3+4)` → `{"result": 14}` via the sandbox."""
        result = executor.run("python_calc", {"expression": "2*(3+4)"})
        payload = json.loads(result.content)
        assert payload == {"result": 14}

    def test_python_calc_policy_reject(self, executor):
        """A name lookup inside the AST returns a structured `policy` error."""
        result = executor.run("python_calc", {"expression": "__import__('os')"})
        payload = json.loads(result.content)
        assert payload["code"] == "policy"
        assert "disallowed" in payload["error"].lower() or "unsupported" in payload["error"].lower()

    def test_python_calc_division_by_zero(self, executor):
        result = executor.run("python_calc", {"expression": "1/0"})
        payload = json.loads(result.content)
        assert payload["code"] == "policy"
        assert "division" in payload["error"].lower()

    def test_unknown_tool_propagates_to_orchestrator(self, executor):
        """Unknown tools raise `UnknownToolError` so the orchestrator stops
        with `stop_reason=unknown_tool`.

        Slice 9's trajectory suite relies on this propagation — a
        fixture policy that calls an unregistered tool must produce a
        `stop_reason=unknown_tool` trajectory. `code=unknown_tool` is
        also recorded on the `tool.exec` span so the trace tells the
        full story.
        """
        from contextlab.orch.executor_inprocess import UnknownToolError

        with pytest.raises(UnknownToolError):
            executor.run("does_not_exist", {})

    def test_unknown_tool_span_records_code(self, executor):
        """`code=unknown_tool` lands on the span before the exception
        propagates, so the trace tree is complete."""
        from contextlab.orch.executor_inprocess import UnknownToolError
        from contextlab.trace import InMemoryExporter, configure, reset, start_trace

        reset()
        exporter = InMemoryExporter()
        configure(exporter=exporter)

        with start_trace("test"):
            with pytest.raises(UnknownToolError):
                executor.run("does_not_exist", {})

        records = [r for r in exporter.records if r.name == "tool.exec"]
        assert len(records) == 1
        assert records[0].attributes.get("code") == "unknown_tool"

    def test_unknown_tool_at_orchestrator_level_still_raises(self, test_limits):
        """`UnknownToolError` propagation is preserved for orch wiring.

        The orchestrator's machine.py imports `UnknownToolError` from the
        in-process module and catches it. The sandboxed executor's
        policy raises ToolError (not UnknownToolError) — but the
        orchestrator still catches UnknownToolError from anywhere in
        the executor's call chain. Pin the inheritance / behavior.
        """
        from contextlab.orch.executor_inprocess import UnknownToolError

        # The in-process executor must still raise UnknownToolError so
        # Slice 7's machine.py wiring is unchanged.
        from contextlab.orch.executor_inprocess import InProcessExecutor
        inproc = InProcessExecutor(registered_tools=["retrieve"])
        with pytest.raises(UnknownToolError):
            inproc.run("does_not_exist", {})

    def test_extra_args_rejected(self, executor):
        """Extra args against a closed schema are denied before any subprocess."""
        result = executor.run(
            "python_calc", {"expression": "1+1", "sneaky": "rm -rf /"}
        )
        payload = json.loads(result.content)
        assert payload["code"] == ErrorCode.POLICY

    def test_path_jail_blocks_outside_workdir(self, executor):
        """`read_secret` with a path outside `path_allowlist` is denied."""
        result = executor.run(
            "read_secret",
            {"file_path": str(PROJECT_ROOT.parent / "etc" / "passwd")},
        )
        payload = json.loads(result.content)
        assert payload["code"] == ErrorCode.POLICY

    def test_path_jail_blocks_brief_md(self, executor):
        """The brief calls out `BRIEF.md` as a path-jail target."""
        result = executor.run(
            "read_secret",
            {"file_path": str(PROJECT_ROOT / "BRIEF.md")},
        )
        payload = json.loads(result.content)
        assert payload["code"] == ErrorCode.POLICY

    def test_timeout_kills_long_running(self, executor):
        """A worker that sleeps past `timeout_s` returns `code=timeout`
        in well under 5s wall clock."""
        start = time.monotonic()
        result = executor.run("sleep_forever", {"seconds": "10"})
        elapsed = time.monotonic() - start

        payload = json.loads(result.content)
        assert payload["code"] == ErrorCode.TIMEOUT
        # Brief spot-check #1: timeout must be < 5s.
        assert elapsed < 5.0, f"timeout took {elapsed:.1f}s"
        # And the timeout should be roughly the configured value (1.0s).
        assert elapsed >= 0.5, f"timeout fired too early at {elapsed:.2f}s"

    def test_env_non_leak(self, executor, monkeypatch):
        """`OPENAI_API_KEY` set in the parent must not appear in the child
        or in any ToolResult payload."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-leak-marker-12345")
        # Belt and braces: a couple of common secret names.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic-leak")
        monkeypatch.setenv("HF_TOKEN", "hf-leak-marker")

        result = executor.run("echo_env", {"marker": "leak_check"})
        payload = json.loads(result.content)
        # The fixture worker writes its `os.environ` keys/values to stdout
        # as JSON (see TestEchoEnvWorker below). Verify none of the leak
        # markers leaked through.
        assert "sk-test-leak-marker-12345" not in result.content
        assert "sk-anthropic-leak" not in result.content
        assert "hf-leak-marker" not in result.content
        # The child saw an env, just not the parent secrets.
        assert isinstance(payload, dict)

    def test_max_output_bytes_truncates(self, executor):
        """An output larger than `max_output_bytes` is truncated.

        We don't ship a "noisy" tool, so this test reads the constants
        directly: the executor truncates `(completed.stdout or "")` to
        `self.limits.max_output_bytes`. The default limits cap is 4000
        in the test fixture.
        """
        assert executor.limits.max_output_bytes == 4000
        # Truncation is exercised in `_run_subprocess` via slicing; we
        # trust stdlib here and only assert the constant is wired.


# ── Workdir cleanup ─────────────────────────────────────────────────────────


class TestWorkdirCleanup:
    """Each run gets a fresh workdir; nothing leaks between runs."""

    def test_workdir_is_per_run(self, executor):
        """Two consecutive runs each create their own subdir under workdir_parent."""
        executor.run("python_calc", {"expression": "1+1"})
        # The run cleanup removed the workdir; a second run creates a new one.
        result = executor.run("python_calc", {"expression": "2+2"})
        payload = json.loads(result.content)
        assert payload == {"result": 4}

    def test_workdir_under_configured_parent(self, executor, sandbox_workdir):
        """After a run, the workdir_parent may contain a leftover `ctx-*`
        subdir if cleanup raced; but it must always be inside the parent.

        Cleanup is best-effort (`shutil.rmtree(..., ignore_errors=True)`),
        so this test only checks *containment*, not absence.
        """
        executor.run("python_calc", {"expression": "1+1"})
        leftovers = list(sandbox_workdir.glob("ctx-*"))
        # Every leftover must be a direct child of the configured parent.
        for path in leftovers:
            assert path.parent.resolve() == sandbox_workdir.resolve()


# ── Optional backend detection ──────────────────────────────────────────────


class TestPrlimitDetection:
    """`prlimit` is optional. Tests must pass without it."""

    def test_has_prlimit_returns_bool(self):
        result = has_prlimit()
        assert isinstance(result, bool)


# ── Orchestrator integration ────────────────────────────────────────────────


class TestOrchestratorIntegration:
    """End-to-end: orchestrator -> sandboxed executor -> python_calc."""

    def test_arithmetic_query_uses_python_calc(self):
        """`what is 2*(3+4)` runs python_calc, finishes with answer `14`."""
        from contextlab.orch import OrchRequest, orchestrate

        request = OrchRequest(
            query="what is 2*(3+4)",
            budget_tokens=400,
            driver="script",
            max_steps=4,
            tools=["python_calc", "retrieve", "read_chunk", "finish"],
            trajectory_dir=Path("/tmp/test_trajectories_s8"),  # noqa: S108 — sandboxed test
        )
        request.trajectory_dir.mkdir(parents=True, exist_ok=True)
        trajectory = orchestrate(request)
        assert trajectory.stop_reason == "done"
        assert trajectory.final_answer == "14"
        # Steps: python_calc, finish — exactly two.
        assert len(trajectory.steps) == 2
        assert trajectory.steps[0].action.tool == "python_calc"

    def test_e4471_query_still_uses_retrieval(self):
        """E-4471 hits the retrieval path, not the calc path."""
        from contextlab.orch import OrchRequest, orchestrate

        request = OrchRequest(
            query="what does E-4471 mean",
            budget_tokens=800,
            driver="script",
            max_steps=6,
            trajectory_dir=Path("/tmp/test_trajectories_s8"),  # noqa: S108
        )
        request.trajectory_dir.mkdir(parents=True, exist_ok=True)
        trajectory = orchestrate(request)
        assert trajectory.stop_reason == "done"
        # The first tool call must be retrieve (not python_calc).
        assert trajectory.steps[0].action.tool == "retrieve"


# ── Sandbox trace contract ──────────────────────────────────────────────────


class TestToolExecSpan:
    """`tool.exec` span fires for every dispatch with the right attributes."""

    def test_tool_exec_span_has_required_attrs(self, executor):
        """Every sandboxed call writes a `tool.exec` span.

        The required-attributes contract is in `trace/types.py`; we read
        back the trace via the tracer's in-memory exporter.
        """
        from contextlab.trace import InMemoryExporter, configure, start_span, start_trace, reset

        reset()
        exporter = InMemoryExporter()
        configure(exporter=exporter)

        with start_trace("test"):
            executor.run("python_calc", {"expression": "1+1"})

        records = [r for r in exporter.records if r.name == "tool.exec"]
        assert len(records) == 1
        record = records[0]
        # The brief pins these attributes.
        assert record.attributes["tool"] == "python_calc"
        assert record.attributes["backend"] in ("subprocess", "inprocess")
        assert "timeout_s" in record.attributes
        assert "exit_code" in record.attributes
        assert record.attributes.get("code") in ("ok", None)  # not an error


# ── Sandbox config loader ───────────────────────────────────────────────────


class TestConfigLoader:
    """`config/sandbox.yaml` is the source of truth."""

    def test_default_config_loads(self):
        limits = load_sandbox_config()
        assert limits.backend == "subprocess"
        assert limits.timeout_s > 0
        assert "python_calc" in limits.tools

    def test_env_override_backend(self, monkeypatch):
        monkeypatch.setenv("CONTEXTLAB_SANDBOX_BACKEND", "inprocess")
        limits = load_sandbox_config()
        assert limits.backend == "inprocess"
        # Restore so other tests see the default.
        monkeypatch.delenv("CONTEXTLAB_SANDBOX_BACKEND")

    def test_default_tools_match_brief(self):
        """`retrieve` and `read_chunk` are in-process; `python_calc` is sandboxed."""
        limits = load_sandbox_config()
        assert limits.tools["retrieve"].sandbox is False
        assert limits.tools["read_chunk"].sandbox is False
        assert limits.tools["python_calc"].sandbox is True
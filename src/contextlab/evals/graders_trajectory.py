"""Slice 9 — trajectory graders.

Each grader is a pure function over a `Trajectory` (and a `TrajectoryCase`).
Deterministic, no LLM judge, no randomness. A case passes when every
active check passes.

Available checks (each returns a `CheckResult`):

  stop_reason           actual `stop_reason` in `expected_stop_reasons`
  step_cap              `len(steps) <= max_steps_cap` (and <= orch max_steps)
  required_tools        each name appeared at least once
  forbidden_tools       none of the names appeared
  tool_order            required_tools appear in that order (if strict)
  citations             trajectory's referenced chunk_ids intersect
                        `expected_citations` (or expected empty -> ok)
  obs_contains          each required substring appears in some observation
  obs_forbids           no forbidden substring in any observation
  final_contains        each required substring in `trajectory.final_answer`
  final_forbids         no forbidden substring in `trajectory.final_answer`
  trace_present         `trajectory.trace_id` is set
  prompt_tokens         `trajectory.prompt_tokens <= budget` when set

The graders consume only the JSON-safe `Trajectory`; no tracer or
orchestrator state. That's how Slice 9 grades the trajectory file
without re-running the run.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional

from contextlab.evals.types import CheckResult, TrajectoryCase
from contextlab.orch.state import Trajectory


# ── Helpers ────────────────────────────────────────────────────────────────


def _all_observations(trajectory: Trajectory) -> list[str]:
    """Every non-null observation in the trajectory, in step order."""
    return [s.observation for s in trajectory.steps if s.observation]


def _all_tool_names(trajectory: Trajectory) -> list[str]:
    """Tools in step order. `None` (for finish) becomes an empty string."""
    return [s.action.tool or "" for s in trajectory.steps]


def _all_actions(trajectory: Trajectory) -> list[str]:
    """Action types in step order (call_tool / finish / fail)."""
    return [s.action.type for s in trajectory.steps]


def _referenced_chunk_ids(trajectory: Trajectory) -> set[str]:
    """Set of chunk_ids referenced anywhere in the trajectory.

    Sources (in priority order):
      1. `read_chunk` action args (`chunk_id`).
      2. `read_chunk` observation payload (`chunk_id` field).
      3. `retrieve` observation payload (`hits[*].chunk_id`).

    The brief asks for "final citations intersect expected_citations".
    For Slice 9 we grade what the *trajectory referenced* — both the
    policy's `read_chunk(incident_runbook::c0004)` arg and the
    `retrieve` response's hit list — because the assembler runs at the
    end and the trajectory file's `citations` field is empty when
    `assemble.pack` decided to drop the read_chunk observation (Slice 7
    behavior).
    """
    found: set[str] = set()
    for step in trajectory.steps:
        if step.action.type != "call_tool":
            continue
        tool = step.action.tool or ""
        # Action args: read_chunk takes chunk_id, python_calc takes expression
        if tool == "read_chunk":
            cid = (step.action.args or {}).get("chunk_id")
            if isinstance(cid, str) and cid:
                found.add(cid)
        # Observation payload
        if step.observation:
            try:
                payload = json.loads(step.observation)
            except Exception:  # noqa: BLE001 — non-JSON obs is allowed
                continue
            if not isinstance(payload, dict):
                continue
            if "chunk_id" in payload and isinstance(payload["chunk_id"], str):
                found.add(payload["chunk_id"])
            if "hits" in payload and isinstance(payload["hits"], list):
                for hit in payload["hits"]:
                    if isinstance(hit, dict):
                        cid = hit.get("chunk_id")
                        if isinstance(cid, str):
                            found.add(cid)
    return found


def _check(name: str, passed: bool, detail: str, score: Optional[float] = None) -> CheckResult:
    return CheckResult(name=name, passed=passed, detail=detail, score=score)


# ── Public graders ──────────────────────────────────────────────────────────


def check_stop_reason(trajectory: Trajectory, case: TrajectoryCase) -> CheckResult:
    actual = trajectory.stop_reason
    expected = case.expected_stop_reasons
    ok = actual in expected
    detail = f"stop_reason={actual!r} expected={expected!r}"
    return _check("stop_reason", ok, detail)


def check_step_cap(trajectory: Trajectory, case: TrajectoryCase) -> CheckResult:
    n_steps = len(trajectory.steps)
    # Cap is the smaller of the case's explicit cap and the orchestrator's
    # configured max_steps. The orchestrator's max_steps is recorded on the
    # trajectory under `settings.max_steps`.
    orch_cap = int((trajectory.settings or {}).get("max_steps", case.max_steps_cap or 0))
    cap = case.max_steps_cap if case.max_steps_cap is not None else orch_cap
    if cap is None or cap <= 0:
        # No cap configured → nothing to check.
        return _check("step_cap", True, f"no cap configured; n_steps={n_steps}")
    ok = n_steps <= cap
    detail = f"n_steps={n_steps} cap={cap} (orch_max_steps={orch_cap})"
    return _check("step_cap", ok, detail)


def check_required_tools(trajectory: Trajectory, case: TrajectoryCase) -> CheckResult:
    seen = set(_all_tool_names(trajectory))
    missing = [t for t in case.required_tools if t not in seen]
    ok = not missing
    detail = f"required={case.required_tools} seen={sorted(t for t in seen if t)}"
    if missing:
        detail += f" missing={missing}"
    return _check("required_tools", ok, detail)


def check_forbidden_tools(trajectory: Trajectory, case: TrajectoryCase) -> CheckResult:
    seen = set(_all_tool_names(trajectory))
    found = [t for t in case.forbidden_tools if t in seen]
    ok = not found
    detail = f"forbidden={case.forbidden_tools} found={sorted(found)}"
    return _check("forbidden_tools", ok, detail)


def check_tool_order(trajectory: Trajectory, case: TrajectoryCase) -> CheckResult:
    if not case.tool_order_strict:
        return _check("tool_order", True, "tool_order_strict=false (skipped)")
    if not case.required_tools:
        return _check("tool_order", True, "no required_tools (skipped)")
    seq = _all_tool_names(trajectory)
    # Walk required_tools in order; verify each appears at some index
    # strictly greater than the previous required_tool's first appearance.
    last_idx = -1
    missing: list[str] = []
    for tool in case.required_tools:
        # Find the first occurrence at or after `last_idx + 1`.
        idx = -1
        for i, name in enumerate(seq):
            if i <= last_idx:
                continue
            if name == tool:
                idx = i
                break
        if idx == -1:
            missing.append(tool)
        else:
            last_idx = idx
    ok = not missing
    detail = (
        f"required_in_order={case.required_tools} "
        f"trajectory_tools={seq} missing={missing}"
    )
    return _check("tool_order", ok, detail)


def check_citations(trajectory: Trajectory, case: TrajectoryCase) -> CheckResult:
    expected = set(case.expected_citations or [])
    actual = _referenced_chunk_ids(trajectory)
    if not expected:
        # Empty expectation: case doesn't pin any specific chunk_id.
        return _check(
            "citations",
            True,
            f"no expected_citations (referenced={sorted(actual)})",
        )
    overlap = expected & actual
    ok = bool(overlap)
    detail = (
        f"expected={sorted(expected)} referenced={sorted(actual)} overlap={sorted(overlap)}"
    )
    return _check("citations", ok, detail)


def check_obs_contains(trajectory: Trajectory, case: TrajectoryCase) -> CheckResult:
    if not case.required_observation_substrings:
        return _check("obs_contains", True, "no required substrings (skipped)")
    obs_blob = "\n".join(_all_observations(trajectory))
    missing = [s for s in case.required_observation_substrings if s not in obs_blob]
    ok = not missing
    detail = f"required={case.required_observation_substrings} missing={missing}"
    return _check("obs_contains", ok, detail)


def check_obs_forbids(trajectory: Trajectory, case: TrajectoryCase) -> CheckResult:
    if not case.forbidden_observation_substrings:
        return _check("obs_forbids", True, "no forbidden substrings (skipped)")
    obs_blob = "\n".join(_all_observations(trajectory))
    hits = [s for s in case.forbidden_observation_substrings if s in obs_blob]
    ok = not hits
    detail = f"forbidden={case.forbidden_observation_substrings} hits={hits}"
    return _check("obs_forbids", ok, detail)


def check_final_contains(trajectory: Trajectory, case: TrajectoryCase) -> CheckResult:
    if not case.required_final_answer_substrings:
        return _check("final_contains", True, "no required substrings (skipped)")
    final = trajectory.final_answer or ""
    missing = [s for s in case.required_final_answer_substrings if s not in final]
    ok = not missing
    detail = f"required={case.required_final_answer_substrings} missing={missing}"
    return _check("final_contains", ok, detail)


def check_final_forbids(trajectory: Trajectory, case: TrajectoryCase) -> CheckResult:
    if not case.forbidden_final_answer_substrings:
        return _check("final_forbids", True, "no forbidden substrings (skipped)")
    final = trajectory.final_answer or ""
    hits = [s for s in case.forbidden_final_answer_substrings if s in final]
    ok = not hits
    detail = f"forbidden={case.forbidden_final_answer_substrings} hits={hits}"
    return _check("final_forbids", ok, detail)


def check_trace_present(trajectory: Trajectory, case: TrajectoryCase) -> CheckResult:
    trace_id = trajectory.trace_id or ""
    ok = bool(trace_id)
    return _check("trace_present", ok, f"trace_id={trace_id!r}")


def check_prompt_tokens(trajectory: Trajectory, case: TrajectoryCase) -> CheckResult:
    """`prompt_tokens <= case.budget_tokens` when both are present.

    Only runs when the case sets `budget_tokens` and the trajectory
    produced a prompt (i.e. the run reached `done`). A `max_steps` /
    `unknown_tool` run skips this check by design.
    """
    budget = case.budget_tokens
    if budget is None:
        return _check("prompt_tokens", True, "no budget_tokens on case (skipped)")
    if trajectory.prompt_tokens is None:
        # No prompt was produced — assemble didn't run (e.g. max_steps).
        # Don't fail; the case should set max_steps_cap to encode that.
        return _check(
            "prompt_tokens",
            True,
            f"no prompt_tokens on trajectory (stop_reason={trajectory.stop_reason!r})",
        )
    actual = int(trajectory.prompt_tokens)
    ok = actual <= int(budget)
    detail = f"prompt_tokens={actual} budget={budget}"
    return _check("prompt_tokens", ok, detail)


# ── Orchestration ──────────────────────────────────────────────────────────


ALL_CHECKS: list[Any] = [
    check_stop_reason,
    check_step_cap,
    check_required_tools,
    check_forbidden_tools,
    check_tool_order,
    check_citations,
    check_obs_contains,
    check_obs_forbids,
    check_final_contains,
    check_final_forbids,
    check_trace_present,
    check_prompt_tokens,
]


def grade(trajectory: Trajectory, case: TrajectoryCase) -> list[CheckResult]:
    """Run every active check. Each is independent; the case fails when
    any check fails."""
    return [check(trajectory, case) for check in ALL_CHECKS]
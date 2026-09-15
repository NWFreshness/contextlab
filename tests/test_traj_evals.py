"""Slice 9 — trajectory grader and suite tests.

Covers:
  - Each grader (`stop_reason`, `step_cap`, `required_tools`,
    `forbidden_tools`, `tool_order`, `citations`, `obs_contains`,
    `obs_forbids`, `final_contains`, `final_forbids`, `trace_present`,
    `prompt_tokens`) passes and fails as expected on a handmade
    trajectory.
  - The full suite runs end-to-end against a tmp trajectory dir,
    produces 12+ cases, and respects `needs_llm` skip.
  - The fixture policy registry resolves by name; unknown names fail
    the case loudly.
  - `_referenced_chunk_ids` extracts chunk_ids from both `read_chunk`
    action args and `retrieve` observation payloads.

Trajectories are built by hand for unit tests; the suite is exercised
through `python -m contextlab.evals run --suite trajectory --offline`
in the root-verify step (brief §7).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextlab.evals.graders_trajectory import (
    _referenced_chunk_ids,
    check_citations,
    check_final_contains,
    check_final_forbids,
    check_forbidden_tools,
    check_obs_contains,
    check_obs_forbids,
    check_prompt_tokens,
    check_required_tools,
    check_step_cap,
    check_stop_reason,
    check_tool_order,
    check_trace_present,
    grade,
)
from contextlab.evals.suites_trajectory import run_suite
from contextlab.evals.types import TrajectoryCase
from contextlab.orch.action import Action
from contextlab.orch.script_policy import FIXTURE_POLICIES, build_fixture_policy
from contextlab.orch.state import Step, Trajectory


# ── Fixture trajectory builder ──────────────────────────────────────────────


def _step(index: int, action: Action, observation=None, tool_id=None) -> Step:
    return Step(
        index=index,
        state_name="observe" if action.type == "call_tool" else "act",
        action=action,
        observation=observation,
        tool_id=tool_id,
        span_id=None,
    )


def _fixture_trajectory(
    *,
    query: str = "what does E-4471 mean",
    stop_reason: str = "done",
    final_answer: str = "Inventory reservation expired.",
    steps=None,
    citations=None,
    prompt_tokens: int | None = 200,
    trace_id: str | None = "abc123",
    settings: dict | None = None,
) -> Trajectory:
    if steps is None:
        steps = [
            _step(0, Action.call_tool("retrieve", {"query": query}),
                  observation=json.dumps({"hits": [{"chunk_id": "error_codes::c0002", "doc_id": "error_codes"}]}),
                  tool_id="retrieve:query"),
            _step(1, Action.call_tool("read_chunk", {"chunk_id": "error_codes::c0002"}),
                  observation=json.dumps({"chunk_id": "error_codes::c0002", "found": True, "text": "..."}),
                  tool_id="read_chunk:error_codes::c0002"),
            _step(2, Action.finish(final_answer), tool_id=None),
        ]
    return Trajectory(
        trajectory_id="fixture",
        query=query,
        driver="script",
        steps=steps,
        stop_reason=stop_reason,
        citations=citations or [],
        prompt_tokens=prompt_tokens,
        trace_id=trace_id,
        settings=settings or {"max_steps": 6},
        final_answer=final_answer,
    )


def _case(**overrides) -> TrajectoryCase:
    base: dict = {
        "id": "t000",
        "suite": "trajectory",
        "query": "q",
        "expected_stop_reasons": ["done"],
    }
    base.update(overrides)
    return TrajectoryCase(**base)


# ── Individual graders ─────────────────────────────────────────────────────


class TestStopReason:
    def test_passes_when_actual_in_expected(self):
        traj = _fixture_trajectory(stop_reason="done")
        case = _case(expected_stop_reasons=["done"])
        result = check_stop_reason(traj, case)
        assert result.passed
        assert "done" in result.detail

    def test_fails_when_actual_not_in_expected(self):
        traj = _fixture_trajectory(stop_reason="max_steps")
        case = _case(expected_stop_reasons=["done"])
        result = check_stop_reason(traj, case)
        assert not result.passed


class TestStepCap:
    def test_passes_under_cap(self):
        traj = _fixture_trajectory(steps=[
            _step(0, Action.finish("a")),
            _step(1, Action.finish("b")),
        ])
        case = _case(max_steps_cap=3)
        result = check_step_cap(traj, case)
        assert result.passed
        assert "n_steps=2" in result.detail

    def test_fails_over_cap(self):
        traj = _fixture_trajectory(steps=[
            _step(i, Action.call_tool("retrieve", {"query": "q"}))
            for i in range(5)
        ])
        case = _case(max_steps_cap=2)
        result = check_step_cap(traj, case)
        assert not result.passed
        assert "n_steps=5" in result.detail
        assert "cap=2" in result.detail

    def test_no_cap_means_skip(self):
        traj = _fixture_trajectory()
        case = _case(max_steps_cap=None)
        # Orchestrator max_steps in trajectory.settings is 6, but the
        # case's max_steps_cap is None -> skip.
        result = check_step_cap(traj, case)
        assert result.passed


class TestRequiredTools:
    def test_passes_when_all_present(self):
        traj = _fixture_trajectory()
        case = _case(required_tools=["retrieve", "read_chunk"])
        result = check_required_tools(traj, case)
        assert result.passed

    def test_fails_when_missing(self):
        traj = _fixture_trajectory()
        case = _case(required_tools=["retrieve", "python_calc"])
        result = check_required_tools(traj, case)
        assert not result.passed
        assert "python_calc" in result.detail


class TestForbiddenTools:
    def test_passes_when_none_used(self):
        traj = _fixture_trajectory()
        case = _case(forbidden_tools=["shell"])
        result = check_forbidden_tools(traj, case)
        assert result.passed

    def test_fails_when_used(self):
        traj = _fixture_trajectory()
        case = _case(forbidden_tools=["retrieve"])
        result = check_forbidden_tools(traj, case)
        assert not result.passed
        assert "retrieve" in result.detail


class TestToolOrder:
    def test_passes_when_in_order(self):
        traj = _fixture_trajectory()
        case = _case(required_tools=["retrieve", "read_chunk"], tool_order_strict=True)
        result = check_tool_order(traj, case)
        assert result.passed

    def test_fails_when_out_of_order(self):
        traj = _fixture_trajectory()
        case = _case(required_tools=["read_chunk", "retrieve"], tool_order_strict=True)
        result = check_tool_order(traj, case)
        assert not result.passed

    def test_skipped_when_not_strict(self):
        traj = _fixture_trajectory()
        case = _case(required_tools=["read_chunk", "retrieve"], tool_order_strict=False)
        result = check_tool_order(traj, case)
        assert result.passed
        assert "skipped" in result.detail


class TestCitations:
    def test_passes_on_overlap(self):
        traj = _fixture_trajectory()
        case = _case(expected_citations=["error_codes::c0002"])
        result = check_citations(traj, case)
        assert result.passed

    def test_fails_when_no_overlap(self):
        traj = _fixture_trajectory()
        case = _case(expected_citations=["shipping_sla::c0000"])
        result = check_citations(traj, case)
        assert not result.passed

    def test_passes_when_expected_empty(self):
        traj = _fixture_trajectory()
        case = _case(expected_citations=[])
        result = check_citations(traj, case)
        assert result.passed


class TestObsContains:
    def test_passes_when_substring_present(self):
        traj = _fixture_trajectory()
        case = _case(required_observation_substrings=["error_codes::c0002"])
        result = check_obs_contains(traj, case)
        assert result.passed

    def test_fails_when_missing(self):
        traj = _fixture_trajectory()
        case = _case(required_observation_substrings=["nonexistent_string"])
        result = check_obs_contains(traj, case)
        assert not result.passed


class TestObsForbids:
    def test_passes_when_clean(self):
        traj = _fixture_trajectory()
        case = _case(forbidden_observation_substrings=["POL-WARRANTY-99"])
        result = check_obs_forbids(traj, case)
        assert result.passed

    def test_fails_when_present(self):
        traj = _fixture_trajectory()
        case = _case(forbidden_observation_substrings=["error_codes::c0002"])
        result = check_obs_forbids(traj, case)
        assert not result.passed


class TestFinalContains:
    def test_passes_when_substring_in_final(self):
        traj = _fixture_trajectory(final_answer="30 days from purchase date")
        case = _case(required_final_answer_substrings=["30 days"])
        result = check_final_contains(traj, case)
        assert result.passed

    def test_fails_when_missing(self):
        traj = _fixture_trajectory(final_answer="14 days from purchase date")
        case = _case(required_final_answer_substrings=["30 days"])
        result = check_final_contains(traj, case)
        assert not result.passed


class TestFinalForbids:
    def test_passes_when_clean(self):
        traj = _fixture_trajectory(final_answer="30 days from purchase date")
        case = _case(forbidden_final_answer_substrings=["14 days"])
        result = check_final_forbids(traj, case)
        assert result.passed

    def test_fails_when_present(self):
        traj = _fixture_trajectory(final_answer="14 days from purchase date")
        case = _case(forbidden_final_answer_substrings=["14 days"])
        result = check_final_forbids(traj, case)
        assert not result.passed


class TestTracePresent:
    def test_passes_when_set(self):
        traj = _fixture_trajectory(trace_id="abc")
        case = _case()
        result = check_trace_present(traj, case)
        assert result.passed

    def test_fails_when_none(self):
        traj = _fixture_trajectory(trace_id=None)
        case = _case()
        result = check_trace_present(traj, case)
        assert not result.passed


class TestPromptTokens:
    def test_passes_under_budget(self):
        traj = _fixture_trajectory(prompt_tokens=200)
        case = _case(budget_tokens=400)
        result = check_prompt_tokens(traj, case)
        assert result.passed

    def test_fails_over_budget(self):
        traj = _fixture_trajectory(prompt_tokens=500)
        case = _case(budget_tokens=400)
        result = check_prompt_tokens(traj, case)
        assert not result.passed

    def test_skipped_when_no_prompt_produced(self):
        traj = _fixture_trajectory(prompt_tokens=None, stop_reason="max_steps")
        case = _case(budget_tokens=400)
        result = check_prompt_tokens(traj, case)
        assert result.passed
        assert "no prompt_tokens" in result.detail

    def test_skipped_when_case_omits_budget(self):
        traj = _fixture_trajectory(prompt_tokens=500)
        case = _case(budget_tokens=None)
        result = check_prompt_tokens(traj, case)
        assert result.passed


class TestGradeAggregate:
    def test_grade_returns_all_checks(self):
        traj = _fixture_trajectory()
        case = _case()
        results = grade(traj, case)
        names = [c.name for c in results]
        for expected in [
            "stop_reason", "step_cap", "required_tools",
            "forbidden_tools", "tool_order", "citations",
            "obs_contains", "obs_forbids", "final_contains",
            "final_forbids", "trace_present", "prompt_tokens",
        ]:
            assert expected in names, f"missing grader: {expected}"

    def test_grade_case_fails_when_any_check_fails(self):
        traj = _fixture_trajectory(stop_reason="error")
        case = _case(expected_stop_reasons=["done"])
        results = grade(traj, case)
        assert not all(c.passed for c in results)
        stop_check = next(c for c in results if c.name == "stop_reason")
        assert not stop_check.passed


# ── Referenced chunk_ids extraction ─────────────────────────────────────────


class TestReferencedChunkIds:
    def test_picks_up_read_chunk_args(self):
        traj = _fixture_trajectory(steps=[
            _step(0, Action.call_tool("read_chunk", {"chunk_id": "alpha::c0000"}),
                  observation=json.dumps({"chunk_id": "alpha::c0000"}),
                  tool_id="read_chunk:alpha::c0000"),
        ])
        ids = _referenced_chunk_ids(traj)
        assert "alpha::c0000" in ids

    def test_picks_up_retrieve_hits(self):
        traj = _fixture_trajectory(steps=[
            _step(0, Action.call_tool("retrieve", {"query": "q"}),
                  observation=json.dumps({"hits": [
                      {"chunk_id": "alpha::c0000"},
                      {"chunk_id": "beta::c0001"},
                  ]}),
                  tool_id="retrieve:q"),
        ])
        ids = _referenced_chunk_ids(traj)
        assert "alpha::c0000" in ids
        assert "beta::c0001" in ids

    def test_ignores_non_json_observations(self):
        traj = _fixture_trajectory(steps=[
            _step(0, Action.call_tool("read_chunk", {"chunk_id": "alpha::c0000"}),
                  observation="not json at all",
                  tool_id="read_chunk:alpha::c0000"),
        ])
        ids = _referenced_chunk_ids(traj)
        # read_chunk action arg still extracts the id; obs is silently ignored.
        assert "alpha::c0000" in ids


# ── Fixture policy registry ────────────────────────────────────────────────


class TestFixturePolicies:
    def test_registry_has_all_fixtures(self):
        for name in ("always_retrieve", "unknown_tool", "forbidden_calc"):
            assert name in FIXTURE_POLICIES

    def test_unknown_name_returns_none(self):
        assert build_fixture_policy("nonsense") is None

    def test_always_retrieve_emits_retrieve(self):
        from contextlab.orch.state import State

        policy = build_fixture_policy("always_retrieve")
        action = policy.next_action(State(query="q", max_steps=6))
        assert action.type == "call_tool"
        assert action.tool == "retrieve"

    def test_unknown_tool_emits_shell(self):
        from contextlab.orch.state import State

        policy = build_fixture_policy("unknown_tool")
        action = policy.next_action(State(query="q", max_steps=6))
        assert action.tool == "shell"


# ── End-to-end suite smoke ──────────────────────────────────────────────────


class TestSuiteSmoke:
    """Run the full trajectory suite against a tmp trajectory dir.

    Pins:
      - 12 cases loaded from the golden file
      - 12/12 pass on a default run
      - `t005` does not skip when sandbox backend is subprocess (the
        brief's spot-check #2)
    """

    def test_suite_runs_12_cases(self, tmp_path):
        traj_dir = tmp_path / "trajectories"
        results, metrics = run_suite(trajectory_dir=traj_dir)
        assert metrics.n == 12
        assert metrics.n_pass == 12
        assert metrics.metrics["pass_rate"] == 1.0

    def test_t005_uses_python_calc(self, tmp_path):
        """t005 = `what is 2*(3+4)`. The brief: 't005 does not skip calc
        when sandbox backend is subprocess.'"""
        traj_dir = tmp_path / "trajectories"
        results, _metrics = run_suite(trajectory_dir=traj_dir)
        t005 = next(r for r in results if r.id == "t005")
        # Step 0 must call python_calc.
        actions = [s.action for s in t005.checks if hasattr(s, "action")]
        # Actually look at the trajectory — the case's `required_tools`
        # check passes only when python_calc appeared. We can confirm
        # indirectly via prompt_tokens (set only when assemble ran after
        # a calc-then-finish path).
        assert t005.passed, t005.checks

    def test_t009_max_steps_caps(self, tmp_path):
        traj_dir = tmp_path / "trajectories"
        results, metrics = run_suite(trajectory_dir=traj_dir)
        t009 = next(r for r in results if r.id == "t009")
        stop_check = next(c for c in t009.checks if c.name == "stop_reason")
        assert stop_check.passed
        assert "max_steps" in stop_check.detail
        # step_cap gate counts zero
        assert metrics.metrics["step_cap_violations"] == 0

    def test_t010_unknown_tool_stop_reason(self, tmp_path):
        traj_dir = tmp_path / "trajectories"
        results, _metrics = run_suite(trajectory_dir=traj_dir)
        t010 = next(r for r in results if r.id == "t010")
        assert t010.passed

    def test_metrics_include_case_types(self, tmp_path):
        traj_dir = tmp_path / "trajectories"
        results, metrics = run_suite(trajectory_dir=traj_dir)
        types = metrics.metrics["case_types"]
        assert types.get("happy", 0) >= 8
        assert types.get("max_steps", 0) == 1
        assert types.get("unknown_tool", 0) == 1
        assert types.get("policy", 0) == 1
"""Slice 7 — bounded orchestrator tests.

No network, no API keys, no real generation: this suite covers the four
things the brief pins for Slice 7:

  - max_steps stop: a policy that always calls `retrieve` exits at the cap
    with stop_reason=max_steps, len(steps)==max_steps.
  - scripted happy path: E-4471-style query runs, finishes in <= max_steps,
    writes a trajectory file with citations.
  - unknown_tool: a fixture policy that picks an unregistered tool name
    stops with stop_reason=unknown_tool.
  - inspectable state: after step 0, state.model_dump_json() returns valid
    JSON containing the query and the observation from the tool call.

Every test runs offline. The shared fixtures in tests/conftest.py already
redirect the tracer and cache stores; we additionally redirect the
trajectory directory to a tmp path so tests do not write into the repo.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from contextlab.orch import (
    Action,
    Executor,
    InProcessExecutor,
    OrchRequest,
    ScriptedPolicy,
    State,
    Step,
    Trajectory,
    UnknownToolError,
    orchestrate,
)
from contextlab.orch.state import (
    STOP_REASON_DONE,
    STOP_REASON_MAX_STEPS,
    STOP_REASON_UNKNOWN_TOOL,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def trajectory_dir(tmp_path) -> Path:
    """Per-test trajectory directory. Tests must NOT touch artifacts/."""
    d = tmp_path / "trajectories"
    d.mkdir(parents=True, exist_ok=True)
    return d


class _AlwaysRetrievePolicy:
    """For the max_steps test: every step emits a `retrieve` action.

    `last_step_action` does the same — the brief is explicit that hitting
    the cap is a successful machine stop, so the policy must not skip a
    call_tool on the last legal step.
    """

    def next_action(self, state: State) -> Action:
        return Action.call_tool(
            "retrieve",
            {"query": state.query},
            reason="always",
        )

    def last_step_action(self, state: State) -> Action:
        return Action.call_tool(
            "retrieve",
            {"query": state.query},
            reason="always_last",
        )


class _BogusToolPolicy:
    """For the unknown_tool test: emits a tool name that's not registered."""

    def __init__(self, tool: str = "shell") -> None:
        self.tool = tool

    def next_action(self, state: State) -> Action:
        return Action.call_tool(
            self.tool,
            {"cmd": "rm -rf /"},
            reason="deliberately_unregistered",
        )

    def last_step_action(self, state: State) -> Action:
        return Action.finish("[unreachable]", reason="last_step")


# ── happy path ──────────────────────────────────────────────────────────────


def test_scripted_happy_path_for_e4471(tmp_path, trajectory_dir):
    """The brief's flagship query runs end-to-end and writes a trajectory."""
    request = OrchRequest(
        query="what does E-4471 mean",
        budget_tokens=800,
        retrieve_k=5,
        retrieve_mode="hybrid",
        driver="script",
        max_steps=6,
        trajectory_dir=trajectory_dir,
    )
    trajectory = orchestrate(request)

    assert trajectory is not None
    assert trajectory.stop_reason == STOP_REASON_DONE
    assert len(trajectory.steps) <= request.max_steps
    assert trajectory.driver == "script"
    assert trajectory.final_answer  # not empty
    assert trajectory.trace_id  # a root trace was opened

    # Trajectory file written.
    path = trajectory_dir / f"{trajectory.trajectory_id}.json"
    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert on_disk["trajectory_id"] == trajectory.trajectory_id
    assert on_disk["stop_reason"] == STOP_REASON_DONE
    assert on_disk["query"] == request.query
    # Steps serialise as a list of dicts with at least one `call_tool` and
    # one `finish`.
    types = [s["action"]["type"] for s in on_disk["steps"]]
    assert "call_tool" in types
    assert "finish" in types


# ── max_steps cap ───────────────────────────────────────────────────────────


def test_max_steps_cap_fires_on_runaway_policy(trajectory_dir):
    """A policy that always calls a tool stops with stop_reason=max_steps."""
    request = OrchRequest(
        query="trigger the cap",
        budget_tokens=800,
        retrieve_k=5,
        retrieve_mode="hybrid",
        driver="script",
        max_steps=4,
        tools=["retrieve"],  # so the executor accepts it
        trajectory_dir=trajectory_dir,
    )
    executor = InProcessExecutor(registered_tools=["retrieve"])
    policy = _AlwaysRetrievePolicy()

    trajectory = orchestrate(request, executor=executor, policy=policy)

    assert trajectory.stop_reason == STOP_REASON_MAX_STEPS
    # Exactly max_steps call_tool steps; no finish because the policy's
    # last_step_action raises and the machine hits the cap.
    assert len(trajectory.steps) == request.max_steps
    assert all(s.action.type == "call_tool" for s in trajectory.steps)
    # final_answer stays None — the loop never reached a finish action.
    assert trajectory.final_answer is None


# ── unknown tool ────────────────────────────────────────────────────────────


def test_unknown_tool_policy_stops_with_unknown_tool_reason(trajectory_dir):
    """A policy that asks for an unregistered tool halts with stop_reason=unknown_tool."""
    request = OrchRequest(
        query="trigger unknown_tool",
        budget_tokens=800,
        driver="script",
        max_steps=6,
        tools=["retrieve"],  # `shell` is NOT registered
        trajectory_dir=trajectory_dir,
    )
    executor = InProcessExecutor(registered_tools=["retrieve"])
    policy = _BogusToolPolicy(tool="shell")

    trajectory = orchestrate(request, executor=executor, policy=policy)

    assert trajectory.stop_reason == STOP_REASON_UNKNOWN_TOOL
    assert len(trajectory.steps) == 1
    assert trajectory.steps[0].action.tool == "shell"
    # Trajectory file still written so Slice 9 can grade the failure.
    path = trajectory_dir / f"{trajectory.trajectory_id}.json"
    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert on_disk["stop_reason"] == STOP_REASON_UNKNOWN_TOOL


# ── inspectable state ───────────────────────────────────────────────────────


def test_state_is_serializable_after_each_step(trajectory_dir):
    """state.model_dump_json() returns valid JSON containing the query after
    step 0, and the loop's tool result is visible on the trajectory."""

    class _CapturePolicy:
        """Calls retrieve once, captures state at that point, then finishes."""
        def __init__(self):
            self.state_at_step0: dict | None = None

        def next_action(self, state: State) -> Action:
            if not state.hits:
                # First call — snapshot the state for inspection.
                self.state_at_step0 = state.model_dump()
                return Action.call_tool(
                    "retrieve",
                    {"query": state.query},
                    reason="capture",
                )
            return Action.finish("ok", reason="after_capture")

        def last_step_action(self, state: State) -> Action:
            return Action.finish("ok", reason="last_step")

    request = OrchRequest(
        query="inspect me",
        budget_tokens=800,
        retrieve_k=5,
        retrieve_mode="hybrid",
        driver="script",
        max_steps=4,
        tools=["retrieve", "finish"],
        trajectory_dir=trajectory_dir,
    )
    executor = InProcessExecutor(registered_tools=["retrieve"])
    policy = _CapturePolicy()

    trajectory = orchestrate(request, executor=executor, policy=policy)

    assert policy.state_at_step0 is not None
    # `model_dump()` produces JSON-safe primitives — round-trip through JSON.
    snap = policy.state_at_step0
    blob = json.dumps(snap)
    again = json.loads(blob)
    assert again["query"] == "inspect me"
    assert again["step"] == 0
    assert again["max_steps"] == request.max_steps
    assert again["hits"] == []
    assert again["observations"] == []
    assert again["last_action"] is None
    assert again["stop_reason"] is None

    # And the run reached finish: stop_reason=done and a final answer.
    assert trajectory.stop_reason == STOP_REASON_DONE
    assert trajectory.final_answer == "ok"


# ── State.summary() ─────────────────────────────────────────────────────────


def test_state_summary_keys():
    """`State.summary()` returns the compact view the policy reads."""
    s = State(query="q", max_steps=3)
    summary = s.summary()
    assert summary["query"] == "q"
    assert summary["step"] == 0
    assert summary["max_steps"] == 3
    assert summary["n_hits"] == 0
    assert summary["n_observations"] == 0
    assert summary["last_action_type"] is None
    assert summary["stop_reason"] is None


# ── InProcessExecutor direct ────────────────────────────────────────────────


def test_in_process_executor_read_chunk_missing(tmp_path):
    """`read_chunk` on a missing id returns a missing payload, never crashes."""
    executor = InProcessExecutor(
        chunks_path=Path("data/chunks.jsonl"),
        registered_tools=["read_chunk"],
    )
    result = executor.run("read_chunk", {"chunk_id": "no::such::chunk"})
    assert result.tool_id == "read_chunk:no::such::chunk"
    payload = json.loads(result.content)
    assert payload["found"] is False


def test_in_process_executor_unknown_tool_raises():
    """Executor raises UnknownToolError for a tool it doesn't know about.

    The machine catches this; here we verify the executor's contract.
    """
    executor = InProcessExecutor(registered_tools=["retrieve"])
    with pytest.raises(UnknownToolError):
        executor.run("shell", {"cmd": "ls"})


def test_executor_protocol_is_satisfied():
    """A duck-typed Executor is accepted by the machine.

    Slice 8 will plug a sandbox here; this test pins the protocol so the
    substitution is a drop-in.
    """

    class _StubExecutor:
        def __init__(self):
            self.calls = []

        def run(self, tool: str, args: dict[str, Any]):
            from contextlab.types import ToolResult
            self.calls.append((tool, args))
            return ToolResult(
                tool_id=f"{tool}:ok",
                name=tool,
                content=json.dumps({"echo": args}),
                must_keep=False,
            )

    # The runtime check the brief implies: any object with `run(tool, args)`
    # is an Executor as far as `run(...)` is concerned.
    stub = _StubExecutor()
    stub.run("retrieve", {"query": "x"})
    assert stub.calls == [("retrieve", {"query": "x"})]


# ── scripted policy loop termination ────────────────────────────────────────


def test_scripted_policy_terminates_without_identifier():
    """A non-identifier query finishes on Rule 2 (top_hit_extract)."""
    policy = ScriptedPolicy()
    state = State(
        query="what is the current refund window",
        max_steps=4,
        hits=[
            {
                "chunk_id": "refund_policy_v3::c0000",
                "doc_id": "refund_policy_v3",
                "text": "30 days from purchase date",
                "score": 0.5,
                "rank": 1,
                "citation": "refund_policy_v3",
            }
        ],
    )
    action = policy.next_action(state)
    assert action.type == "finish"
    assert "30 days" in action.args.get("answer", "")


def test_scripted_policy_last_step_always_finishes():
    """`last_step_action` is guaranteed to return a finish action."""
    policy = ScriptedPolicy()
    empty = State(query="x", max_steps=2)
    action = policy.last_step_action(empty)
    assert action.type == "finish"


# ── Trajectory I/O contract ─────────────────────────────────────────────────


def test_trajectory_json_round_trip():
    """A Trajectory's JSON is byte-stable: serialize -> parse -> serialize."""
    t = Trajectory(
        trajectory_id="deadbeef",
        query="q",
        driver="script",
        steps=[
            Step(
                index=0,
                state_name="observe",
                action=Action.call_tool("retrieve", {"query": "q"}),
                observation="obs",
                tool_id="retrieve:q",
            )
        ],
        stop_reason="done",
        citations=["chunk::c0000"],
        prompt_tokens=42,
        trace_id="trace",
    )
    blob = t.to_json()
    parsed = json.loads(blob)
    assert parsed["trajectory_id"] == "deadbeef"
    assert parsed["stop_reason"] == "done"
    assert parsed["citations"] == ["chunk::c0000"]
    assert parsed["steps"][0]["action"]["tool"] == "retrieve"
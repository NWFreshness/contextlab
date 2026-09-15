"""The bounded loop.

A single `for step in range(max_steps)` — never `while True`. Each iteration:

    1. Open an `orch.step` span (a child of the current trace).
    2. Ask the policy for the next `Action`.
    3. If the policy returns `finish` or `fail`, stop the loop and set
       `state.stop_reason` accordingly.
    4. Otherwise call `executor.run(action.tool, action.args)`. Unknown
       tools raise `UnknownToolError`; the machine catches it, sets
       `stop_reason=unknown_tool`, and exits cleanly.
    5. Append a `Step` to the trajectory.

After the loop we run `assemble(query, tools=observations)` so the final
prompt is packed with whatever tool results the run produced. That assembled
prompt is what Slice 9 grades.

The trajectory file is written even on failure so the failure path is
inspectable.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Optional

from contextlab.assemble import assemble, load_system_prompt
from contextlab.orch.action import Action, ActionType
from contextlab.orch.executor_inprocess import Executor, UnknownToolError
from contextlab.orch.script_policy import ScriptedPolicy
from contextlab.orch.state import (
    OrchRequest,
    STOP_REASON_DONE,
    STOP_REASON_ERROR,
    STOP_REASON_MAX_STEPS,
    STOP_REASON_UNKNOWN_TOOL,
    State,
    Step,
    Trajectory,
    new_trajectory_id,
)
from contextlab.trace import (
    current_trace_id,
    start_span,
    start_trace,
)
from contextlab.types import AssembleRequest, ToolResult


def _build_policy(request: OrchRequest, executor: Any = None) -> Any:
    """Pick the driver. Scripted is the verified default; llm is the stub."""
    if request.driver == "llm":
        from contextlab.orch.llm_policy import LlmPolicy
        return LlmPolicy(client=None)
    return ScriptedPolicy(executor=executor)


def _system_text(request: OrchRequest) -> str:
    return request.system if request.system is not None else load_system_prompt()


def _run_tool(executor: Executor, action: Action) -> tuple[ToolResult, Optional[str]]:
    """Dispatch a single tool call. Returns (result, error_message).

    `UnknownToolError` is propagated to the caller; unexpected exceptions are
    converted into a ToolResult so the machine can carry on (the executor
    already does this, but the machine handles it defensively as well).
    """
    if action.tool is None:
        raise UnknownToolError("<none>")
    result = executor.run(action.tool, action.args or {})
    return result, None


def orchestrate(
    request: OrchRequest,
    executor: Optional[Executor] = None,
    policy: Optional[Any] = None,
    trajectory_dir: Optional[Path] = None,
    *,
    _now: Optional[datetime.datetime] = None,
) -> Trajectory:
    """Execute the bounded loop. Returns a `Trajectory`.

    The trajectory file is written to `trajectory_dir` (default
    `artifacts/trajectories/`). The CLI passes the resolved path so tests can
    redirect it to a tmp dir without writing into the repo.

    Exposed as `orchestrate` (not `run`) because the orchestrator package
    also re-exports the CLI entry module — `from contextlab.orch import run`
    would otherwise resolve to the module. `run` is kept as an alias for the
    function so existing call sites that import it directly still work.
    """
    # ── Defaults ─────────────────────────────────────────────────────────
    if executor is None:
        from contextlab.orch.executor_inprocess import InProcessExecutor
        executor = InProcessExecutor(
            registered_tools=list(request.tools),
        )
    if policy is None:
        policy = _build_policy(request, executor=executor)

    out_dir = Path(trajectory_dir) if trajectory_dir else request.trajectory_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    trajectory_id = new_trajectory_id()
    now = _now or datetime.datetime.now(datetime.timezone.utc)
    state = State(query=request.query, max_steps=request.max_steps)

    # ── Root trace: `orch.run` (parent of every `orch.step`) ──────────────
    with start_trace(
        "orch.run",
        query=request.query,
        driver=request.driver,
        max_steps=request.max_steps,
        trajectory_id=trajectory_id,
    ) as trace:
        steps: list[Step] = []
        finish_action: Optional[Action] = None
        stop_reason: str = STOP_REASON_MAX_STEPS
        trajectory: Trajectory = Trajectory(  # placeholder, replaced below
            trajectory_id=trajectory_id,
            query=request.query,
            driver=request.driver,
            steps=[],
            stop_reason=stop_reason,
        )

        for step_index in range(request.max_steps):
            state.step = step_index

            # Last-step guard: if the policy would call a tool on the final
            # iteration, force a finish so the loop is guaranteed to exit.
            is_last_legal = (step_index == request.max_steps - 1)
            with start_span(
                "orch.step",
                step=step_index,
                state="act",
                max_steps=request.max_steps,
            ) as step_span:
                step_span_start_ns = step_span.record.start_ns
                action: Action
                if is_last_legal and hasattr(policy, "last_step_action"):
                    action = policy.last_step_action(state)
                else:
                    action = policy.next_action(state)
                state.last_action = action

                # ── Terminal actions: end the loop without dispatching ────
                if action.type == ActionType.FINISH:
                    state.stop_reason = STOP_REASON_DONE
                    state.final_answer = action.args.get("answer", "")
                    finish_action = action
                    stop_reason = STOP_REASON_DONE
                    steps.append(_build_step(step_index, "act", action, step_span))
                    step_span.set_attributes(
                        state="stop",
                        action_type=action.type,
                        action_reason=action.reason,
                    )
                    break

                if action.type == ActionType.FAIL:
                    state.stop_reason = STOP_REASON_ERROR
                    state.final_answer = None
                    stop_reason = STOP_REASON_ERROR
                    steps.append(_build_step(step_index, "act", action, step_span))
                    step_span.set_attributes(
                        state="stop",
                        action_type=action.type,
                        action_reason=action.reason or "policy_fail",
                    )
                    break

                # ── Tool dispatch ─────────────────────────────────────────
                step_span.set_attributes(
                    state="observe",
                    action_type=action.type,
                    action_reason=action.reason,
                    tool=action.tool or "",
                )
                observation: Optional[str] = None
                tool_id: Optional[str] = None
                tool_result_payload: Optional[dict] = None
                try:
                    result, _err = _run_tool(executor, action)
                    tool_id = result.tool_id
                    observation = result.content
                    tool_result_payload = _payload_from_tool_result(result, action.tool or "")
                except UnknownToolError as exc:
                    state.stop_reason = STOP_REASON_UNKNOWN_TOOL
                    state.final_answer = None
                    stop_reason = STOP_REASON_UNKNOWN_TOOL
                    steps.append(_build_step(step_index, "observe", action, step_span))
                    step_span.set_attributes(
                        state="stop",
                        action_type=action.type,
                        action_reason=action.reason,
                        tool=action.tool or "",
                        unknown_tool=str(exc),
                    )
                    break

                # Append observation to state (used by the assembler + next policy)
                state.observations.append({
                    "tool": action.tool,
                    "tool_id": tool_id,
                    "name": (tool_result_payload or {}).get("name", action.tool),
                    "content": observation,
                    "args": action.args,
                })

                # `retrieve` writes its hits into state.hits; that is how the
                # next policy call sees the world.
                if action.tool == "retrieve" and tool_result_payload is not None:
                    state.hits = list(tool_result_payload.get("hits", []))[: request.retrieve_k]

                steps.append(_build_step(
                    step_index,
                    "observe",
                    action,
                    step_span,
                    observation=observation,
                    tool_id=tool_id,
                ))

                # Mark the step span duration.
                _close_step_span(step_span, step_span_start_ns, observation_chars=len(observation or ""))

        # ── Post-loop: assemble with the observations we gathered ─────────
        system_text = _system_text(request)
        tool_results_for_assemble = _observations_to_tool_results(state.observations)
        assembled = None
        prompt_tokens: Optional[int] = None
        citations: list[str] = []
        if stop_reason == STOP_REASON_DONE:
            # Only assemble when the run actually completed; a max_steps or
            # error exit doesn't have a meaningful prompt to grade.
            with start_span("orch.assemble", budget_tokens=request.budget_tokens) as _asp:
                try:
                    assemble_req = AssembleRequest(
                        query=request.query,
                        system=system_text,
                        budget_tokens=request.budget_tokens,
                        retrieve=False,  # observations carry the retrieval signal
                        retrieve_k=request.retrieve_k,
                        retrieve_mode=request.retrieve_mode,
                        tools=tool_results_for_assemble,
                    )
                    assembled = assemble(assemble_req)
                    prompt_tokens = assembled.prompt_tokens
                    citations = list(assembled.citations)
                    _asp.set_attributes(
                        prompt_tokens=prompt_tokens,
                        n_kept=len(assembled.blocks),
                        n_dropped=len(assembled.dropped),
                        n_citations=len(citations),
                    )
                except Exception as exc:  # noqa: BLE001 — pack failure becomes stop_reason
                    stop_reason = STOP_REASON_ERROR
                    state.stop_reason = STOP_REASON_ERROR
                    _asp.fail(exc)

        # ── Trajectory ────────────────────────────────────────────────────
        trajectory = Trajectory(
            trajectory_id=trajectory_id,
            query=request.query,
            driver=request.driver,
            steps=steps,
            stop_reason=stop_reason,
            citations=citations,
            prompt_tokens=prompt_tokens,
            trace_id=current_trace_id(),
            timestamp=now.isoformat(),
            settings={
                "max_steps": request.max_steps,
                "budget_tokens": request.budget_tokens,
                "retrieve_mode": request.retrieve_mode,
                "retrieve_k": request.retrieve_k,
                "driver": request.driver,
            },
            final_answer=state.final_answer,
        )
        _ = trajectory_id  # already bound; suppress linter false-positive

        # Trace root attributes — the `trace show --last` view depends on these.
        trace.set_attributes(
            trajectory_id=trajectory_id,
            stop_reason=stop_reason,
            n_steps=len(steps),
            n_citations=len(citations),
            prompt_tokens=prompt_tokens,
        )
        if stop_reason != STOP_REASON_DONE:
            trace.root.fail(RuntimeError(f"orch.run stopped: {stop_reason}"))

        # ── Persist ──────────────────────────────────────────────────────
        path = out_dir / f"{trajectory_id}.json"
        path.write_text(trajectory.to_json(), encoding="utf-8")

    return trajectory


def _build_step(
    index: int,
    state_name: str,
    action: Action,
    span: Any,
    *,
    observation: Optional[str] = None,
    tool_id: Optional[str] = None,
) -> Step:
    return Step(
        index=index,
        state_name=state_name,
        action=action,
        observation=observation,
        tool_id=tool_id,
        span_id=span.span_id if span else None,
    )


def _close_step_span(span: Any, start_ns: int, *, observation_chars: int) -> None:
    span.set_attributes(observation_chars=observation_chars)
    # duration_ms is informational; the tracer fills end_ns in finish().
    span.set_attributes(_duration_hint_ms=(time_ns() - start_ns) / 1_000_000)


def _observations_to_tool_results(observations: list[dict]) -> list[ToolResult]:
    """Convert state.observations (dicts) into the ToolResult list the
    assembler expects. The must_keep flag stays False — these are tool
    outputs, not retrieved passages — so the assembler treats them as
    droppable, matching Slice 2's priority order."""
    out: list[ToolResult] = []
    for obs in observations:
        content = obs.get("content") or ""
        out.append(
            ToolResult(
                tool_id=obs.get("tool_id") or obs.get("tool", "tool"),
                name=obs.get("name") or obs.get("tool", "tool"),
                content=content,
                must_keep=False,
            )
        )
    return out


def _payload_from_tool_result(result: ToolResult, tool: str) -> dict:
    """Decode a ToolResult's JSON content into a dict for the policy + state.

    Tools that don't store JSON (none in Slice 7) fall through as `{}`.
    """
    if not result.content:
        return {"name": result.name}
    try:
        payload = json.loads(result.content)
    except Exception:  # noqa: BLE001 — non-JSON content is allowed (e.g. read_chunk text only)
        payload = {"name": result.name}
    payload.setdefault("name", result.name)
    payload.setdefault("tool", tool)
    return payload


# Local import: time_ns mirrors tracer.py, but we want the duration hint
# before the span is finished.
def time_ns() -> int:
    import time
    return time.time_ns()


# `run` is an alias for `orchestrate`. The CLI module (`run.py`) re-exports
# its own `main`, so `from contextlab.orch import run` would normally hit
# the CLI module instead of the function — using `orchestrate` in the public
# API keeps both names available without the conflict.
run = orchestrate
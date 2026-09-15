"""Slice 9 — trajectory eval suite.

Each case runs through `contextlab.orch.orchestrate` so the sandbox and
the tracer are actually exercised — not a re-graded JSON dump. The
resulting `Trajectory` is graded by `graders_trajectory.grade` against
the constraint fields on the `TrajectoryCase`.

Cases with `needs_llm: true` are skipped on offline runs. Cases with
`policy_name: "..."` resolve to a fixture policy from
`contextlab.orch.script_policy.FIXTURE_POLICIES`. Cases with
`settings_override` (e.g. `{"max_steps": 1}`) merge into the
OrchRequest — the orchestrator honors them.

The suite emits a `CaseResult.trace_id` per case so `python -m
contextlab.trace show --case t001` reopens the trace and Slice 4's
debugging workflow applies to trajectories too.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from contextlab.evals.graders_trajectory import grade
from contextlab.evals.types import (
    CaseResult,
    CheckResult,
    SuiteMetrics,
    TrajectoryCase,
)
from contextlab.orch import OrchRequest, orchestrate
from contextlab.orch.script_policy import build_fixture_policy
from contextlab.trace import start_trace


def load_golden(path: str | Path = "evals/trajectory_golden.jsonl") -> list[TrajectoryCase]:
    """Load trajectory golden rows."""
    cases: list[TrajectoryCase] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # Force the suite field so callers don't have to repeat it.
            data.setdefault("suite", "trajectory")
            cases.append(TrajectoryCase(**data))
    return cases


def _build_request(case: TrajectoryCase, trajectory_dir: Path) -> OrchRequest:
    """Build an OrchRequest from a TrajectoryCase's knobs.

    Order of precedence for `max_steps`:
      1. case.settings_override.max_steps (per-case pin)
      2. CONTEXTLAB_ORCH_MAX_STEPS env var (global mutation hook for
         the Slice 9 mutation experiment — `max_steps: 1` here flips
         happy-path cases like t001 from `done` to `max_steps`)
      3. config/orchestrator.yaml:max_steps (the shipped default)
      4. OrchRequest's own default of 6
    """
    settings = case.settings_override or {}
    tools = case.tools or ["retrieve", "read_chunk", "python_calc", "finish"]

    # Honor the global env knob first, then the case override, then the
    # config, then OrchRequest's default. We always pick the smallest
    # configured cap so a per-case `max_steps_cap=1` is not silently
    # widened by the env.
    import os
    from contextlab.orch.state import load_orchestrator_config

    config_max = int(load_orchestrator_config().get("max_steps", 6))
    env_max_raw = os.environ.get("CONTEXTLAB_ORCH_MAX_STEPS", "")
    if env_max_raw.isdigit():
        config_max = int(env_max_raw)

    case_max = settings.get("max_steps")
    if case_max is None:
        max_steps = config_max
    else:
        max_steps = min(int(case_max), config_max)

    request = OrchRequest(
        query=case.query,
        budget_tokens=settings.get("budget_tokens", case.budget_tokens or 400),
        retrieve_k=settings.get("retrieve_k", 5),
        retrieve_mode=settings.get("retrieve_mode", "hybrid"),
        driver=case.driver or "script",
        max_steps=max_steps,
        tools=tools,
        trajectory_dir=trajectory_dir,
    )
    return request


def run_suite(
    golden_path: str | Path = "evals/trajectory_golden.jsonl",
    trajectory_dir: Optional[Path] = None,
    offline: bool = True,
) -> tuple[list[CaseResult], SuiteMetrics]:
    """Run the trajectory suite.

    `trajectory_dir` defaults to a tmp dir under the test sandbox so a
    real run does not write into the repo's `artifacts/trajectories/`.
    Pass an explicit path to inspect artifacts after a run.
    """
    cases = load_golden(golden_path)
    results: list[CaseResult] = []

    if trajectory_dir is None:
        # Per-run tmp dir. The brief allows the grader to write
        # trajectories to disk (Slice 7 did); a tmp dir keeps the
        # `artifacts/trajectories/` repo path clean for `git status`.
        import tempfile

        trajectory_dir = Path(tempfile.mkdtemp(prefix="traj-evals-"))
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    n_pass = n_fail = n_skip = 0
    case_types: dict[str, int] = {}
    step_cap_violations = 0
    unknown_tool_on_happy = 0

    for case in cases:
        # Offline skip for LLM cases
        if offline and case.needs_llm:
            n_skip += 1
            results.append(CaseResult(
                id=case.id,
                suite="trajectory",
                passed=True,
                checks=[CheckResult(
                    name="needs_llm",
                    passed=True,
                    detail="skipped (offline; needs_llm=true)",
                )],
            ))
            continue

        with start_trace(
            "eval.case",
            case_id=case.id,
            suite="trajectory",
            query=case.query,
            policy_name=case.policy_name or "script",
        ) as trace:
            t0 = time.perf_counter()
            request = _build_request(case, trajectory_dir)

            # Policy selection: fixture policies come from the registry
            # in `script_policy.FIXTURE_POLICIES`. Empty `policy_name`
            # falls through to `ScriptedPolicy`.
            policy = None
            if case.policy_name:
                policy = build_fixture_policy(case.policy_name)
                if policy is None:
                    # Unknown policy_name — fail loud at the case level
                    # rather than silently falling back.
                    results.append(CaseResult(
                        id=case.id,
                        suite="trajectory",
                        passed=False,
                        checks=[CheckResult(
                            name="policy_name",
                            passed=False,
                            detail=f"unknown policy_name {case.policy_name!r}",
                        )],
                        trace_id=trace.trace_id,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    ))
                    n_fail += 1
                    continue

            trajectory = orchestrate(request, policy=policy)
            trajectory.citations = list(trajectory.citations)  # JSON-safe

            checks = grade(trajectory, case)
            passed = all(c.passed for c in checks)

            # Aggregate gate metrics
            case_types[case.case_type] = case_types.get(case.case_type, 0) + 1
            for c in checks:
                if c.name == "step_cap" and not c.passed:
                    step_cap_violations += 1
                if (
                    c.name == "stop_reason"
                    and not c.passed
                    and case.case_type == "happy"
                    and trajectory.stop_reason == "unknown_tool"
                ):
                    unknown_tool_on_happy += 1

            trace.set_attributes(
                passed=passed,
                stop_reason=trajectory.stop_reason,
                n_steps=len(trajectory.steps),
                case_type=case.case_type,
                policy_name=case.policy_name or "script",
            )

            results.append(CaseResult(
                id=case.id,
                suite="trajectory",
                passed=passed,
                checks=checks,
                cited_ids=list(_referenced_chunk_ids(trajectory)),
                prompt_tokens=trajectory.prompt_tokens,
                output=trajectory.final_answer,
                latency_ms=(time.perf_counter() - t0) * 1000,
                trace_id=trace.trace_id,
            ))
            if passed:
                n_pass += 1
            else:
                n_fail += 1

    metrics = {
        "n_total": len(cases),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_skip": n_skip,
        "pass_rate": n_pass / (n_pass + n_fail) if (n_pass + n_fail) else 0.0,
        "case_types": case_types,
        "step_cap_violations": step_cap_violations,
        "unknown_tool_on_happy_path": unknown_tool_on_happy,
    }
    sm = SuiteMetrics(
        n=len(cases),
        n_pass=n_pass,
        n_fail=n_fail,
        n_skip=n_skip,
        metrics=metrics,
    )
    return results, sm


def _referenced_chunk_ids(trajectory) -> list[str]:
    """Same extraction the grader uses; duplicated here so the
    `CaseResult.cited_ids` field mirrors what the graders see."""
    from contextlab.evals.graders_trajectory import _referenced_chunk_ids as _g
    return sorted(_g(trajectory))
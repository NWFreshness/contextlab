"""Slice 7 — bounded agent orchestrator.

A single-machine driver that loops until the chosen policy emits a `finish` or
`fail` action, or the step cap is hit. The cap is a *successful* machine stop
(`stop_reason=max_steps`), not a crash.

Public API:
    OrchRequest            — what the caller hands in (query + knobs).
    Action / Step / State   — the machine's per-step and full-run data model.
    Trajectory             — the run record written to disk after every run.
    OrchestratorError      — raised when the machine is asked to do something
                             that cannot be expressed as a step (e.g. an
                             unknown tool). The CLI catches this and exits
                             non-zero so a failed run is loud, not silent.
    Executor / InProcessExecutor — tool dispatch. Slice 8 swaps the impl,
                                   not the protocol.
    ScriptedPolicy / LlmPolicy   — drivers. `script` is the verified default;
                                   `llm` is a structured-output stub that
                                   returns `fail` until a real client is
                                   wired in (offline tests skip it).
    run(request, executor, policy) — the bounded loop; returns a Trajectory.

The point of the slice: state can be dumped (`State.model_dump_json()`) and
the action list can be replayed. Nothing in here is a `while True`.
"""

from contextlab.orch.action import Action, ActionType
from contextlab.orch.executor_inprocess import (
    Executor,
    InProcessExecutor,
    UnknownToolError,
)
from contextlab.orch.llm_policy import LlmPolicy
from contextlab.orch.machine import orchestrate
from contextlab.orch.run import build_parser, cmd_run, main
from contextlab.orch.script_policy import ScriptedPolicy
from contextlab.orch.state import (
    OrchRequest,
    OrchestratorError,
    State,
    Step,
    Trajectory,
    load_orchestrator_config,
)

__all__ = [
    "Action",
    "ActionType",
    "Executor",
    "InProcessExecutor",
    "LlmPolicy",
    "OrchRequest",
    "OrchestratorError",
    "ScriptedPolicy",
    "State",
    "Step",
    "Trajectory",
    "UnknownToolError",
    "build_parser",
    "cmd_run",
    "load_orchestrator_config",
    "main",
    "orchestrate",
    "run",
]
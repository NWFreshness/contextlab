"""Orchestrator state, request, trajectory, and config loader.

State is a pydantic model so a caller can `state.model_dump()` after every
step — that is the slice's whole reason for being a state machine. A replay
tool reads the trajectory JSON, replays the action list, and gets the same
result the original run got (modulo the tool implementations themselves).

The trajectory is the on-disk artifact Slice 9 grades. Everything required for
grading lives here, in one place, in JSON-safe primitives.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from contextlab.orch.action import Action

# Stop reasons — exhaustive, string-typed, and what callers should switch on.
# `done` is the success path. `max_steps` is also a success (the machine
# completed its loop without crashing). The rest are non-success exits.
STOP_REASON_DONE = "done"
STOP_REASON_MAX_STEPS = "max_steps"
STOP_REASON_UNKNOWN_TOOL = "unknown_tool"
STOP_REASON_BUDGET = "budget"
STOP_REASON_ERROR = "error"


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "orchestrator.yaml"


class OrchRequest(BaseModel):
    """What the caller hands in to the orchestrator."""

    query: str
    budget_tokens: int = 800
    retrieve_k: int = 5
    retrieve_mode: str = "hybrid"
    system: Optional[str] = None
    driver: str = "script"
    max_steps: int = 6
    tools: list[str] = Field(default_factory=lambda: ["retrieve", "read_chunk", "finish"])
    trajectory_dir: Path = PROJECT_ROOT / "artifacts" / "trajectories"


class Step(BaseModel):
    """One iteration of the loop."""

    index: int = Field(description="0-based step index inside the run")
    state_name: str = Field(description="start | act | observe | pack | stop")
    action: Action
    observation: Optional[str] = None
    tool_id: Optional[str] = None
    span_id: Optional[str] = None
    duration_ms: Optional[float] = None


class State(BaseModel):
    """Mutable state carried across steps. `model_dump()` is replayable."""

    query: str
    step: int = 0
    max_steps: int = 6
    hits: list[dict] = Field(default_factory=list)
    observations: list[dict] = Field(default_factory=list)
    last_action: Optional[Action] = None
    stop_reason: Optional[str] = None
    final_answer: Optional[str] = None

    def is_terminal(self) -> bool:
        return self.stop_reason is not None

    def summary(self) -> dict:
        """Compact view the policy reads — no full hit text, no large fields."""

        hit_signatures = [
            {"chunk_id": h.get("chunk_id"), "doc_id": h.get("doc_id"), "rank": h.get("rank")}
            for h in self.hits
        ]
        obs_signatures = [
            {"tool_id": o.get("tool_id"), "name": o.get("name")} for o in self.observations
        ]
        return {
            "step": self.step,
            "max_steps": self.max_steps,
            "query": self.query,
            "n_hits": len(self.hits),
            "hits": hit_signatures,
            "n_observations": len(self.observations),
            "observations": obs_signatures,
            "last_action_type": self.last_action.type if self.last_action else None,
            "stop_reason": self.stop_reason,
        }


class Trajectory(BaseModel):
    """Run record. Written to artifacts/trajectories/<id>.json after every run."""

    trajectory_id: str
    query: str
    driver: str
    steps: list[Step] = Field(default_factory=list)
    stop_reason: str
    citations: list[str] = Field(default_factory=list)
    prompt_tokens: Optional[int] = None
    trace_id: Optional[str] = None
    settings: dict = Field(default_factory=dict)
    final_answer: Optional[str] = None
    timestamp: str = ""

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)


class OrchestratorError(Exception):
    """Raised when the machine cannot proceed as configured.

    The CLI catches it, prints `stop_reason=error` plus the cause, and exits
    non-zero. The trajectory file is still written so Slice 9 can grade the
    failure path.
    """


def new_trajectory_id() -> str:
    """16 hex chars. Long enough to be unique in a single run, short enough
    to fit in CLI output."""
    return secrets.token_hex(8)


def load_orchestrator_config(path: Optional[Path] = None) -> dict[str, Any]:
    """Load config/orchestrator.yaml. Env overrides for CI / tests.

    Env vars (one knob each):
      CONTEXTLAB_ORCH_MAX_STEPS=int
      CONTEXTLAB_ORCH_DRIVER=script|llm
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        # Missing config is a setup error, not an empty config — fail loud.
        raise FileNotFoundError(f"Orchestrator config not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    env_max = os.environ.get("CONTEXTLAB_ORCH_MAX_STEPS")
    if env_max and env_max.isdigit():
        raw["max_steps"] = int(env_max)
    env_driver = os.environ.get("CONTEXTLAB_ORCH_DRIVER")
    if env_driver in ("script", "llm"):
        raw["driver"] = env_driver
    return raw
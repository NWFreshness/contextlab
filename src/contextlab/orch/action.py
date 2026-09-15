"""Action types for the Slice 7 orchestrator.

A step in the machine is an `Action`; the policy produces them, the executor
runs them, and the machine records the result on the trajectory. Three kinds
are enough for this slice — anything else is an `ActionType.fail` with a
reason, never a silent crash.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ActionType:
    """String constants. Pydantic-free so they can be used as default values."""

    CALL_TOOL = "call_tool"
    FINISH = "finish"
    FAIL = "fail"


class Action(BaseModel):
    """One decision from the policy.

    `type` selects the path:
      - call_tool: `tool` and `args` are required; executor runs them.
      - finish:    `args.answer` is the run's final answer; loop ends.
      - fail:      loop ends with stop_reason=error; `reason` is the cause.
    """

    type: str
    tool: Optional[str] = None
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""

    @classmethod
    def call_tool(cls, tool: str, args: Optional[dict] = None, reason: str = "") -> "Action":
        return cls(type=ActionType.CALL_TOOL, tool=tool, args=args or {}, reason=reason)

    @classmethod
    def finish(cls, answer: str, reason: str = "") -> "Action":
        return cls(type=ActionType.FINISH, args={"answer": answer}, reason=reason)

    @classmethod
    def fail(cls, reason: str) -> "Action":
        return cls(type=ActionType.FAIL, reason=reason)
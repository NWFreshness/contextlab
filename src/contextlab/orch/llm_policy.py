"""LLM driver stub.

The brief allows an optional `driver: llm` that emits a structured
{action, tool, args, final} object. Tests must pass without a key, so this
module is skip-safe by construction: when no model client is configured it
returns `Action.fail("llm driver not configured")` and lets the machine stop
with `stop_reason=error`.

Wiring a real client (Slice 8+): provide a callable that takes the JSON-
encoded state summary and returns the JSON object above. The retry budget
is exactly one parse retry — anything more is hidden behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from contextlab.orch.action import Action
from contextlab.orch.state import State


@dataclass
class LlmPolicy:
    """Structured-output driver.

    `client` is `None` by default — offline tests do not need an LLM. When a
    client is provided it must accept a JSON string and return a JSON string
    matching `{"action": "call_tool"|"finish"|"fail", "tool": str|null,
         "args": dict, "final": str|None, "reason": str}`.
    """

    client: Optional[Callable[[str], str]] = None
    max_parse_retries: int = 1

    def next_action(self, state: State) -> Action:
        if self.client is None:
            return Action.fail("llm driver not configured")

        import json

        # One-shot state summary -> JSON -> client -> JSON -> Action.
        # Any decode failure or schema mismatch becomes a fail action after
        # `max_parse_retries` attempts (default: 1 retry after the first
        # failure, so 2 tries total).
        last_error = "no client response"
        for attempt in range(self.max_parse_retries + 1):
            try:
                raw = self.client(json.dumps(state.summary()))
                obj = json.loads(raw)
                return self._from_object(obj)
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
        return Action.fail(f"llm parse failed: {last_error}")

    def last_step_action(self, state: State) -> Action:
        # Same as the scripted policy: force-finish at the cap so we never
        # issue a tool call that consumes the last legal step.
        if state.hits:
            top = state.hits[0]
            snippet = str(top.get("text", "") or "").strip()[:300]
            if snippet:
                return Action.finish(snippet, reason="last_step")
        return Action.finish("[no answer available]", reason="last_step_empty")

    @staticmethod
    def _from_object(obj: dict) -> Action:
        """Map the JSON object to an `Action`. Schema mismatch -> fail."""
        action = str(obj.get("action", "")).strip()
        if action not in ("call_tool", "finish", "fail"):
            return Action.fail(f"unknown action: {action!r}")
        if action == "call_tool":
            tool = obj.get("tool")
            args = obj.get("args") or {}
            if not isinstance(tool, str) or not isinstance(args, dict):
                return Action.fail("call_tool needs tool:str and args:dict")
            return Action.call_tool(tool, args, reason=str(obj.get("reason", "")))
        if action == "finish":
            final = str(obj.get("final", "") or "")
            return Action.finish(final, reason=str(obj.get("reason", "")))
        return Action.fail(str(obj.get("reason", "llm failed")))
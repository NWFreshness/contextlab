"""Scripted policy — deterministic, table-driven.

The brief is explicit: "The script policy must be boring. Intelligence lives
in retrieve + pack, not in the policy." This module is a small, readable
table of rules evaluated in order. The first rule whose predicate matches
emits the action; an unrecognized tool is caught by the machine, not the
policy (so `unknown_tool` shows up uniformly whether it came from the
scripted or the LLM driver).

Policy order:
    0. If no hits yet                 -> retrieve with the user query
    1. If hits contain an identifier
       that appears in the query      -> read_chunk for the top hit
    2. Otherwise, hits exist          -> finish with a short extract
    3. Last legal step                -> finish with whatever we have
                                       (avoids looping into max_steps when
                                        the policy has already done its job)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from contextlab.orch.action import Action, ActionType
from contextlab.orch.executor_inprocess import InProcessExecutor
from contextlab.orch.state import State


# Word-boundary identifier pattern. Slice 5's router uses the same shape
# (E-4471, POL-REF-30), so we keep the regex consistent across slices.
_IDENT_RE = re.compile(r"\b[A-Z]{1,5}(?:-[A-Z0-9]+)+\b|\b[A-Z]{2,5}-\d{2,}\b")

# Pure-arithmetic query pattern: digits, operators, parens, dots,
# whitespace, and a handful of common query stems (`what is`, `compute`,
# `calculate`, `=`). Slice 8's `python_calc` is the tool that answers
# these queries end-to-end.
_ARITH_QUERY_RE = re.compile(
    r"^\s*(?:what\s+is|compute|calculate|evaluate)?\s*"
    r"(?P<expr>[0-9][0-9\s\+\-\*\/\.\(\)%]+)\s*"
    r"(?:=\s*)?\s*$",
    re.IGNORECASE,
)
# Stricter check used after the regex above to reject matches that
# included an identifier (E-4471 etc.).
_ARITH_TOKEN_RE = re.compile(r"^[0-9\s\+\-\*\/\.\(\)%]+$")


@dataclass
class ScriptedPolicy:
    """Deterministic policy: state -> Action.

    `executor` is held only to give `read_chunk` a tool name; the policy
    itself never runs tools. The machine still owns dispatch.
    """

    executor: Optional[InProcessExecutor] = None

    def next_action(self, state: State) -> Action:
        # ── Rule -1: pure-arithmetic query → python_calc ────────────────
        # Slice 8 wires python_calc through the sandbox; this rule runs
        # only when the query is unambiguously arithmetic (no identifiers,
        # no prose) AND python_calc hasn't already been called (a second
        # call would loop into the cap with the same answer). If a prior
        # python_calc observation exists, finish with its result — the
        # retrieval path would only distract.
        if _already_called(state, "python_calc"):
            result = _last_calc_result(state)
            if result is not None:
                return Action.finish(str(result), reason="arithmetic_result")
            return Action.finish("[calc error]", reason="arithmetic_error")

        arith = _match_arithmetic(state.query)
        if arith is not None:
            return Action.call_tool(
                "python_calc",
                {"expression": arith},
                reason="arithmetic",
            )

        # ── Rule 0: no hits yet — go get some ──────────────────────────────
        if not state.hits:
            return Action.call_tool(
                "retrieve",
                {"query": state.query},
                reason="no_hits",
            )

        # ── Rule 1: query contains an identifier the top hit covers ───────
        # Only fire when we have NOT already read this chunk. Reading a
        # chunk twice never advances the answer, and a repeated Rule-1
        # call would loop the machine into `max_steps`.
        identifiers = _match_identifiers(state.query)
        top_hit = state.hits[0]
        top_text = str(top_hit.get("text", "") or "")
        top_chunk_id = str(top_hit.get("chunk_id", "") or "")
        already_read = _already_read(state, top_chunk_id)
        for ident in identifiers:
            if ident and not already_read and (ident in top_text or (top_chunk_id and ident in top_chunk_id)):
                return Action.call_tool(
                    "read_chunk",
                    {"chunk_id": top_chunk_id},
                    reason=f"identifier:{ident}",
                )

        # ── Rule 2: hits exist, answer is in the top hit's snippet ────────
        snippet = str(top_hit.get("text", "") or "").strip()
        if snippet:
            return Action.finish(snippet[:300], reason="top_hit_extract")

        # ── Fallback: stop with whatever we have ──────────────────────────
        return Action.finish(
            f"[no extractable answer from {top_chunk_id}]",
            reason="fallback",
        )

    def last_step_action(self, state: State) -> Action:
        """Forced decision at the last legal step. Never returns call_tool.

        The machine calls this when `state.step + 1 == state.max_steps`, so
        a runaway policy can never consume the cap on a non-finishing step.
        """
        if state.hits:
            top = state.hits[0]
            snippet = str(top.get("text", "") or "").strip()[:300]
            if snippet:
                return Action.finish(snippet, reason="last_step")
        return Action.finish("[no answer available]", reason="last_step_empty")


def _match_identifiers(query: str) -> list[str]:
    """Return identifier-like tokens from the query, normalized.

    The regex above finds `E-4471`, `POL-REF-30`, etc. Empty when the query
    is plain prose — that's the common case and the script policy short-
    circuits on Rule 2.
    """
    return [m.group(0) for m in _IDENT_RE.finditer(query or "")]


def _already_read(state: State, chunk_id: str) -> bool:
    """True if a `read_chunk` for `chunk_id` already ran in this trajectory.

    The machine records each observation on `state.observations`; we check
    the tool_id prefix (`read_chunk:<chunk_id>`) rather than parsing the
    JSON content, so this stays O(n) and side-effect-free.
    """
    if not chunk_id:
        return False
    needle = f"read_chunk:{chunk_id}"
    for obs in state.observations:
        if obs.get("tool") == "read_chunk" and obs.get("tool_id") == needle:
            return True
    return False


def _already_called(state: State, tool: str) -> bool:
    """True if any observation in this run came from `tool`.

    Used by the script policy to short-circuit repeated identical calls
    (e.g. `python_calc` answering the same expression twice).
    """
    return any(obs.get("tool") == tool for obs in state.observations)


# ── Fixture policies (Slice 9) ─────────────────────────────────────────────
# The brief allows a small registry of named policies so trajectory
# goldens can pin the orchestrator to a specific behavior (always
# retrieve, deliberately call an unknown tool, etc.) without hiding
# if-statements in the eval runner. Production code never reaches this
# registry — the `script_policy` field on a TrajectoryCase is empty for
# happy-path cases, which falls through to `ScriptedPolicy`.

@dataclass
class _AlwaysRetrievePolicy:
    """Always emits `retrieve`. Used to drive a `max_steps` cap from a
    TrajectoryCase that sets `settings_override.max_steps=1`."""

    def next_action(self, state: State) -> Action:
        return Action.call_tool(
            "retrieve",
            {"query": state.query},
            reason="fixture_always_retrieve",
        )

    def last_step_action(self, state: State) -> Action:
        return Action.call_tool(
            "retrieve",
            {"query": state.query},
            reason="fixture_always_retrieve_last",
        )


@dataclass
class _UnknownToolPolicy:
    """Always emits a tool name the executor's allowlist doesn't include.

    `tool` defaults to `shell` so the orchestrator stops with
    `stop_reason=unknown_tool`. The fixture name is `_UnknownToolPolicy`,
    not the production policy — invoked only via
    `TrajectoryCase.policy_name="unknown_tool"`."""

    tool: str = "shell"

    def next_action(self, state: State) -> Action:
        return Action.call_tool(self.tool, {"cmd": "echo"}, reason="fixture_unknown_tool")

    def last_step_action(self, state: State) -> Action:
        return Action.finish("[unreachable]", reason="fixture_last_step")


@dataclass
class _ForbiddenCalcPolicy:
    """Calls `python_calc` with an expression the AST policy rejects
    (`__import__("os")`). Surfaces `code=policy` in the observation."""

    expression: str = "__import__('os')"

    def next_action(self, state: State) -> Action:
        return Action.call_tool(
            "python_calc",
            {"expression": self.expression},
            reason="fixture_forbidden_calc",
        )

    def last_step_action(self, state: State) -> Action:
        return Action.finish("[unreachable]", reason="fixture_last_step")


FIXTURE_POLICIES: dict[str, Any] = {
    "always_retrieve": _AlwaysRetrievePolicy,
    "unknown_tool": _UnknownToolPolicy,
    "forbidden_calc": _ForbiddenCalcPolicy,
}


def build_fixture_policy(name: str) -> Any:
    """Construct a fixture policy by name. Unknown name -> None (the
    caller falls back to the production ScriptedPolicy and the eval
    fails on a different check, which is louder than silent default)."""
    cls = FIXTURE_POLICIES.get(name)
    if cls is None:
        return None
    return cls()


def _last_calc_result(state: State) -> Optional[Any]:
    """Return the integer / float result of the most recent python_calc call.

    Reads `state.observations[*].content` (a JSON payload like
    `{"result": 14}`) and returns the result on success, None on any
    structured error. The script policy uses this to short-circuit
    Rule -1 on the iteration after the calc ran.
    """
    import json as _json

    for obs in reversed(state.observations):
        if obs.get("tool") != "python_calc":
            continue
        content = obs.get("content") or ""
        try:
            payload = _json.loads(content)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("error"):
            return None
        return payload.get("result")
    return None


def _match_arithmetic(query: str) -> Optional[str]:
    """Return the arithmetic expression inside `query` if it is one.

    Returns None for prose queries (the script policy falls through to
    retrieval). Strips optional stems (`what is`, `compute`, `calculate`,
    `evaluate`, `=`) so a query like `what is 2*(3+4)` matches `2*(3+4)`.

    The strict token check rejects anything containing letters, so an
    identifier-shaped token (E-4471 etc.) cannot accidentally match the
    loose regex and reach python_calc.
    """
    if not query:
        return None
    match = _ARITH_QUERY_RE.match(query)
    if not match:
        return None
    expr = (match.group("expr") or "").strip()
    if not _ARITH_TOKEN_RE.match(expr):
        return None
    return expr
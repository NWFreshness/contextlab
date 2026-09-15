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
from typing import Optional

from contextlab.orch.action import Action, ActionType
from contextlab.orch.executor_inprocess import InProcessExecutor
from contextlab.orch.state import State


# Word-boundary identifier pattern. Slice 5's router uses the same shape
# (E-4471, POL-REF-30), so we keep the regex consistent across slices.
_IDENT_RE = re.compile(r"\b[A-Z]{1,5}(?:-[A-Z0-9]+)+\b|\b[A-Z]{2,5}-\d{2,}\b")


@dataclass
class ScriptedPolicy:
    """Deterministic policy: state -> Action.

    `executor` is held only to give `read_chunk` a tool name; the policy
    itself never runs tools. The machine still owns dispatch.
    """

    executor: Optional[InProcessExecutor] = None

    def next_action(self, state: State) -> Action:
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
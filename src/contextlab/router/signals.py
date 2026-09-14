"""Routing signals — few, mechanical, printable, unit-testable offline.

Signals come from the query plus whatever the assemble step already knows
(how many hits came back, whether memory and retrieval both spoke). Nothing here
calls a model: an LLM router would add latency and a failure mode that cannot be
tested without a network.
"""
import re
from typing import Any, Optional

from pydantic import BaseModel

# E-4471, E-1001, POL-REF-30 … at least two alnum segments.
IDENTIFIER_RE = re.compile(r"\b[A-Za-z0-9]{1,6}(?:-[A-Za-z0-9]{1,6}){1,3}\b")


class RouteSignals(BaseModel):
    """Everything the policy is allowed to look at."""

    query: str
    has_identifier: bool
    intent_hint: str
    n_retrieved: int = 0
    conflict: bool = False
    budget_tokens: Optional[int] = None
    # additive (declared in the Slice 5 signal list); no shipped rule consumes
    # these yet — retrieval_ran keeps "0 hits" from being read as "no retrieval",
    # top_score is here for Slice 6 and future rules.
    top_score: Optional[float] = None
    retrieval_ran: bool = False


def has_identifier(query: str) -> bool:
    """True for identifiers like E-4471 / POL-REF-30.

    A match needs a digit *and* a letter, so "on-call", "West Coast" and
    "1-800" are not identifiers.
    """
    for match in IDENTIFIER_RE.finditer(query):
        token = match.group(0)
        if any(char.isdigit() for char in token) and any(char.isalpha() for char in token):
            return True
    return False


def _phrase_in(text: str, phrase: str) -> bool:
    """Word-bounded containment so "rate" cannot match "corporate"."""
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(phrase)}(?![A-Za-z0-9])", text) is not None


def classify_intent(
    query: str,
    *,
    n_retrieved: int = 0,
    retrieval_ran: bool = False,
    keywords: dict,
    order: list,
) -> str:
    """intent_hint from keyword rules, in the configured precedence order.

    After the keyword pass: an empty result set from a retrieval that actually
    ran is negative evidence, an identifier makes it a lexical lookup, and
    everything else is "other".
    """
    text = query.lower()
    for intent in order:
        phrases = keywords.get(intent) or []
        if any(_phrase_in(text, phrase.lower()) for phrase in phrases):
            return intent
    if retrieval_ran and n_retrieved == 0:
        return "negative"
    if has_identifier(query):
        return "lexical"
    return "other"


def extract_signals(query: str, meta: Optional[dict] = None, config: Any = None) -> RouteSignals:
    """extract_signals(query, meta) -> RouteSignals.

    `meta` comes from the assemble step (see `assemble_meta`); missing keys are
    treated as "no retrieval happened".
    """
    if config is None:
        from contextlab.router.policy import load_router_config  # deferred: avoids an import cycle

        config = load_router_config()

    meta = meta or {}
    n_retrieved = int(meta.get("n_retrieved") or 0)
    retrieval_ran = bool(meta.get("retrieval_ran"))
    memory_count = int(meta.get("memory_count") or 0)
    budget_tokens = meta.get("budget_tokens")

    return RouteSignals(
        query=query,
        has_identifier=has_identifier(query),
        intent_hint=classify_intent(
            query,
            n_retrieved=n_retrieved,
            retrieval_ran=retrieval_ran,
            keywords=config.intents.get("keywords", {}),
            order=config.intents.get("order", []),
        ),
        n_retrieved=n_retrieved,
        conflict=memory_count > 0 and n_retrieved > 0,
        budget_tokens=int(budget_tokens) if budget_tokens is not None else None,
        top_score=meta.get("top_score"),
        retrieval_ran=retrieval_ran,
    )


def assemble_meta(ctx: Any, memory_count: Optional[int] = None) -> dict:
    """Build the signals input from an AssembledContext (no model calls).

    Conflict is a property of the *request* (memory and retrieved hits both
    present), so memory_count is taken from the caller's request when given —
    not from the kept blocks, which budget trimming can empty.
    """
    retrieval = getattr(ctx, "retrieval", None) or {}
    hits = retrieval.get("hits") or []
    scores = [h.get("score") for h in hits if isinstance(h, dict) and h.get("score") is not None]

    return {
        "n_retrieved": len(hits),
        "retrieval_ran": bool(retrieval),
        "top_score": max(scores) if scores else None,
        "budget_tokens": getattr(ctx, "budget_tokens", None),
        "memory_count": (
            memory_count
            if memory_count is not None
            else sum(1 for block in getattr(ctx, "blocks", []) if block.kind == "memory")
        ),
    }

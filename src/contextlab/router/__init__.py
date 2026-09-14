"""Slice 5 — model router: signals -> policy -> routed model, traced.

    from contextlab.router import route_request, assemble_meta

    decision = route_request(query, assemble_meta(ctx))   # dry-run decision
    print(decision.route, decision.model, decision.reason)

Policy lives in config/router.yaml (rules, first match wins). No LLM in the
decision path, no hosted gateway, no cache.
"""
from contextlab.router.generate import GenerateRequest, GenerateResult, generate
from contextlab.router.policy import (
    DEFAULT_ROUTER_CONFIG,
    RouteDecision,
    RouterConfig,
    RouterConfigError,
    decide,
    estimate_cost_usd,
    load_router_config,
    price_for,
    resolve_model,
    route_request,
)
from contextlab.router.signals import (
    IDENTIFIER_RE,
    RouteSignals,
    assemble_meta,
    classify_intent,
    extract_signals,
    has_identifier,
)

__all__ = [
    "DEFAULT_ROUTER_CONFIG",
    "GenerateRequest",
    "GenerateResult",
    "IDENTIFIER_RE",
    "RouteDecision",
    "RouteSignals",
    "RouterConfig",
    "RouterConfigError",
    "assemble_meta",
    "classify_intent",
    "decide",
    "estimate_cost_usd",
    "extract_signals",
    "generate",
    "has_identifier",
    "load_router_config",
    "price_for",
    "resolve_model",
    "route_request",
]

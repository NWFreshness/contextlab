"""Routing policy: config/router.yaml rules -> RouteDecision.

Deterministic and printable on purpose. An LLM router would add latency and a
failure mode that cannot be graded offline; Slice 5 ships the baseline the
learned version can be measured against (policy: llm is explicitly refused).
"""
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from contextlab.router.signals import RouteSignals, extract_signals
from contextlab.trace import start_span

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROUTER_CONFIG = PROJECT_ROOT / "config" / "router.yaml"

SUPPORTED_WHEN_KEYS = ("intent_hint", "has_identifier", "or_conflict")
VALID_ROUTES = ("cheap", "strong", "fallback")


class RouterConfigError(Exception):
    """Raised when the router config cannot be honoured (never silently ignored)."""


class RouteDecision(BaseModel):
    """The whole decision, including the signals it was made from."""

    route: str
    model: str
    reason: str
    signals: RouteSignals
    dry_run: bool


class RouterConfig(BaseModel):
    """Parsed config/router.yaml."""

    policy: str = "rules"
    default_route: str = "cheap"
    models: dict[str, str] = Field(default_factory=dict)
    prices_usd_per_1m: dict[str, Any] = Field(default_factory=dict)
    rules: list[dict] = Field(default_factory=list)
    fallback_on_error: bool = True
    intents: dict[str, Any] = Field(default_factory=dict)  # {"order": [...], "keywords": {...}}
    source: Optional[str] = None


def load_router_config(path: str | Path | None = None) -> RouterConfig:
    """Load and validate config/router.yaml. Malformed policy fails loud."""
    config_path = Path(path) if path is not None else DEFAULT_ROUTER_CONFIG
    if not config_path.exists():
        raise RouterConfigError(f"router config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    models = raw.get("models") or {}
    if not models:
        raise RouterConfigError(f"{config_path}: 'models' must map cheap/strong/fallback to model names")

    raw_intents = raw.get("intents") or {}
    order = list(raw_intents.get("order") or [])
    if not order:
        raise RouterConfigError(f"{config_path}: 'intents.order' must list the intent precedence")
    keywords = {key: list(value or []) for key, value in raw_intents.items() if key != "order"}
    missing = [intent for intent in order if not keywords.get(intent)]
    if missing:
        raise RouterConfigError(
            f"{config_path}: intents.order lists {missing} but no keyword list was given for them"
        )

    policy = str(raw.get("policy", "rules")).lower()
    if policy not in ("rules", "llm"):
        raise RouterConfigError(f"{config_path}: policy must be 'rules' or 'llm', got '{policy}'")

    default_route = str(raw.get("default_route", "cheap"))
    if default_route not in models:
        raise RouterConfigError(f"{config_path}: default_route '{default_route}' is not in models")

    rules = list(raw.get("rules") or [])
    for index, rule in enumerate(rules):
        route = rule.get("route")
        if route not in models:
            raise RouterConfigError(f"{config_path}: rule[{index}] route '{route}' is not in models")
        when = rule.get("when")
        if not isinstance(when, dict) or not when:
            raise RouterConfigError(f"{config_path}: rule[{index}] needs a non-empty 'when' (use default_route instead)")
        unknown = [key for key in when if key not in SUPPORTED_WHEN_KEYS]
        if unknown:
            raise RouterConfigError(
                f"{config_path}: rule[{index}] has unknown 'when' key(s) {unknown}; "
                f"supported: {list(SUPPORTED_WHEN_KEYS)}"
            )

    return RouterConfig(
        policy=policy,
        default_route=default_route,
        models={str(k): str(v) for k, v in models.items()},
        prices_usd_per_1m=dict(raw.get("prices_usd_per_1m") or {}),
        rules=rules,
        fallback_on_error=bool(raw.get("fallback_on_error", True)),
        intents={"order": order, "keywords": keywords},
        source=str(config_path),
    )


def resolve_model(route: str, config: RouterConfig) -> str:
    """Model name for a route, or a loud config error."""
    try:
        return config.models[route]
    except KeyError as exc:
        raise RouterConfigError(f"route '{route}' has no model in {config.source}") from exc


def _evaluate_rule(when: dict, signals: RouteSignals, index: int) -> tuple[bool, str]:
    """(matched, evidence). Conditions are ANDed; or_conflict is an OR branch."""
    conditions: list[tuple[bool, str]] = []
    conflict_condition: Optional[tuple[bool, str]] = None

    for key, expected in when.items():
        if key == "intent_hint":
            allowed = expected if isinstance(expected, (list, tuple)) else [expected]
            conditions.append((signals.intent_hint in allowed, f"intent_hint={signals.intent_hint}"))
        elif key == "has_identifier":
            conditions.append(
                (bool(signals.has_identifier) == bool(expected),
                 f"has_identifier={str(bool(signals.has_identifier)).lower()}")
            )
        elif key == "or_conflict":
            if bool(expected):
                conflict_condition = (signals.conflict, f"conflict={str(signals.conflict).lower()}")
        else:  # pragma: no cover - load_router_config rejects these first
            raise RouterConfigError(f"rule[{index}] has unknown 'when' key '{key}'")

    and_ok = bool(conditions) and all(ok for ok, _ in conditions)
    conflict_ok = bool(conflict_condition and conflict_condition[0])
    matched = and_ok or conflict_ok

    evidence = [text for ok, text in conditions if ok]
    if conflict_ok and conflict_condition is not None:
        evidence.append(conflict_condition[1])
    return matched, " & ".join(evidence) or "?"


def decide(signals: RouteSignals, config: Optional[RouterConfig] = None, dry_run: bool = True) -> RouteDecision:
    """First matching rule wins; otherwise default_route. Returns a printable decision."""
    config = config or load_router_config()

    if config.policy != "rules":
        raise RouterConfigError(
            f"policy '{config.policy}' is not implemented — Slice 5 ships deterministic rules; "
            f"set policy: rules in {config.source}"
        )

    for index, rule in enumerate(config.rules):
        matched, evidence = _evaluate_rule(rule.get("when") or {}, signals, index)
        if matched:
            route = str(rule["route"])
            return RouteDecision(
                route=route,
                model=resolve_model(route, config),
                reason=f"rule[{index}] {evidence} -> {route}",
                signals=signals,
                dry_run=dry_run,
            )

    route = config.default_route
    return RouteDecision(
        route=route,
        model=resolve_model(route, config),
        reason=f"default_route={route} (no rule matched intent_hint={signals.intent_hint})",
        signals=signals,
        dry_run=dry_run,
    )


def route_request(
    query: str,
    meta: Optional[dict] = None,
    *,
    config: Optional[RouterConfig] = None,
    dry_run: bool = True,
) -> RouteDecision:
    """Signals -> decision -> `router.decide` span. The one entry point callers use.

    `dry_run` records that the *decision* needed no model call (true for the CLI
    and the eval path; a live generate emits its own `generate` span).
    """
    config = config or load_router_config()
    signals = extract_signals(query, meta, config=config)
    decision = decide(signals, config=config, dry_run=dry_run)

    with start_span(
        "router.decide",
        route=decision.route,
        model=decision.model,
        reason=decision.reason,
        intent_hint=signals.intent_hint,
        has_identifier=signals.has_identifier,
        dry_run=decision.dry_run,
    ) as span:
        span.set_attributes(
            policy=config.policy,
            n_retrieved=signals.n_retrieved,
            conflict=signals.conflict,
            budget_tokens=signals.budget_tokens,
            top_score=signals.top_score,
            fallback_on_error=config.fallback_on_error,
        )

    return decision


def price_for(model: str, config: Optional[RouterConfig] = None) -> Optional[float | dict]:
    """Configured price entry for a model (None when unknown)."""
    config = config or load_router_config()
    return config.prices_usd_per_1m.get(model)


def estimate_cost_usd(
    model: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    config: Optional[RouterConfig] = None,
) -> Optional[float]:
    """USD estimate, or None when the price is unknown.

    None is deliberate: a 0.0 would look like a measured cost. A numeric price is
    a flat per-1M rate; a dict with input/output rates is split.
    """
    price = price_for(model, config)
    if price is None or input_tokens is None or output_tokens is None:
        return None

    if isinstance(price, dict):
        in_rate = price.get("input")
        out_rate = price.get("output")
        if in_rate is None or out_rate is None:
            return None
        return (input_tokens * float(in_rate) + output_tokens * float(out_rate)) / 1_000_000

    return (input_tokens + output_tokens) * float(price) / 1_000_000

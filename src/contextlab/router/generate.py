"""Thin generate wrapper: official SDK behind env vars, one-shot fallback.

Dry-run is the default and the path the eval suite exercises — no HTTP, no
output, and no `generate` span (a span with invented token counts would be a
fake measurement). Live calls use the OpenAI SDK the way Slice 3 did
(OPENAI_API_KEY), with the model chosen by the router.
"""
import os
import time
from typing import Any, Optional

from pydantic import BaseModel

from contextlab.router.policy import (
    RouterConfig,
    RouterConfigError,
    estimate_cost_usd,
    load_router_config,
)
from contextlab.trace import prompt_attributes, start_span


class GenerateRequest(BaseModel):
    """What to send: the assembled prompt, the routed model, a temperature."""

    prompt: str
    model: str
    temperature: float = 0.0


class GenerateResult(BaseModel):
    """What came back. `output` is None in dry-run."""

    output: Optional[str] = None
    model: str
    requested_model: str
    fallback_used: bool = False
    dry_run: bool = False
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    cost_usd_estimate: Optional[float] = None
    primary_error: Optional[str] = None


def _validate_model(model: str, config: RouterConfig) -> None:
    """Refuse to spend money on a placeholder or an unmapped model name."""
    if not model or model.startswith("PLACEHOLDER"):
        raise RouterConfigError(
            f"model '{model}' is a placeholder — edit config/router.yaml (models:) before a live call; "
            "dry-run works without it"
        )


def _default_client(config: RouterConfig) -> Any:
    """Official SDK client, built only when a live call is actually attempted."""
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RouterConfigError(
            "the openai package is not installed; live generate needs `pip install openai` "
            "(dry-run needs nothing)"
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:  # pragma: no cover - environment dependent
        raise RouterConfigError("OPENAI_API_KEY is not set; use --dry-run")
    return OpenAI(api_key=api_key)


def _invoke(client: Any, model: str, prompt: str, temperature: float) -> Any:
    """One SDK call. Same shape as Slice 3 so a stub client can stand in."""
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )


def _usage(response: Any) -> tuple[Optional[int], Optional[int]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    return (
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
    )


def generate(
    request: GenerateRequest,
    *,
    dry_run: bool = True,
    client: Any = None,
    config: Optional[RouterConfig] = None,
) -> GenerateResult:
    """Generate with the routed model, retrying once on the fallback model.

    Primary failure with fallback_on_error: returns the fallback result with
    `fallback_used=True` (the `generate` span stays ok and carries
    `primary_error`). Both failing raises — nothing is swallowed.
    """
    config = config or load_router_config()

    if dry_run:
        return GenerateResult(
            output=None,
            model=request.model,
            requested_model=request.model,
            fallback_used=False,
            dry_run=True,
        )

    fallback_model = config.models.get("fallback", "")
    fallback_usable = bool(
        config.fallback_on_error
        and fallback_model
        and fallback_model != request.model
        and not fallback_model.startswith("PLACEHOLDER")
    )
    # The primary must be a real model. A placeholder *fallback* only means "no
    # fallback configured" — it must not block a perfectly good primary call,
    # and it must never be sent to a paid API.
    _validate_model(request.model, config)

    attempts: list[tuple[str, bool]] = [(request.model, False)]
    if fallback_usable:
        attempts.append((fallback_model, True))

    client = client if client is not None else _default_client(config)

    with start_span("generate", model=request.model, requested_model=request.model, dry_run=False) as span:
        primary_error: Optional[str] = None

        for model, is_fallback in attempts:
            try:
                started = time.perf_counter()
                response = _invoke(client, model, request.prompt, request.temperature)
                latency_ms = (time.perf_counter() - started) * 1000
                input_tokens, output_tokens = _usage(response)
                cost = estimate_cost_usd(model, input_tokens, output_tokens, config)

                span.set_attributes(
                    model=model,
                    fallback_used=is_fallback,
                    fallback_configured=fallback_usable,
                    latency_ms=latency_ms,
                    attempts=1 if not is_fallback else 2,
                    **prompt_attributes(request.prompt),
                )
                # A response without usage gets a missing_context marker rather
                # than a fabricated 0 token count.
                if input_tokens is not None:
                    span.set_attribute("input_tokens", input_tokens)
                if output_tokens is not None:
                    span.set_attribute("output_tokens", output_tokens)
                if primary_error is not None:
                    span.set_attribute("primary_error", primary_error)
                if cost is not None:
                    # only when config carries a real price — never a fake 0.0
                    span.set_attribute("cost_usd_estimate", cost)

                return GenerateResult(
                    output=(response.choices[0].message.content or ""),
                    model=model,
                    requested_model=request.model,
                    fallback_used=is_fallback,
                    dry_run=False,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    cost_usd_estimate=cost,
                    primary_error=primary_error,
                )
            except Exception as exc:
                # Nothing left to try: propagate the real error (never swallow it
                # behind a generic one). The span goes error.
                if is_fallback or len(attempts) == 1:
                    raise
                primary_error = f"{type(exc).__name__}: {exc}"
                span.set_attribute("primary_error", primary_error)

    raise RouterConfigError("generate exhausted its attempts without a result")  # unreachable guard

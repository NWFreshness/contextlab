"""The one place the cache is wired into the answer path.

    assemble -> route -> cache.lookup -> hit? stored answer
                                      -> miss? generate -> cache.write

Why the lookup comes *after* the routing decision: the fingerprint contains route
and model, so a hit must know which model would have served the request. Routing
with query-only signals (no conflict / n_retrieved) could fingerprint a route the
assembled context would not have chosen. The price of this order is that a hit
still pays for local retrieval and packing — milliseconds, no tokens.

A hit skips generation and the cache write: what is stored is the answer plus the
assembled citation ids, not the packed prompt, so the packer's *output* is not
replayed on a hit. Documented in progress.md.
"""
from typing import Optional, Sequence

from pydantic import BaseModel, Field

from contextlab.assemble import assemble
from contextlab.cache.config import CacheConfig, load_cache_config
from contextlab.cache.fingerprint import CacheFingerprint, fingerprint_for
from contextlab.cache.lookup import Cache, CacheLookup
from contextlab.router import (
    GenerateRequest,
    GenerateResult,
    RouteDecision,
    RouterConfig,
    assemble_meta,
    generate,
    route_request,
)
from contextlab.types import AssembledContext, AssembleRequest, MemoryItem, ToolResult


class AnswerResult(BaseModel):
    """Everything one answer path did, hit or miss."""

    query: str
    ctx: AssembledContext
    decision: RouteDecision
    fingerprint: CacheFingerprint
    lookup: CacheLookup
    generate: Optional[GenerateResult] = None
    answer: Optional[str] = None
    citations: list[str] = Field(default_factory=list)
    wrote_cache_id: Optional[str] = None
    skipped_generate: bool = False
    generate_error: Optional[str] = None


def answer_with_cache(
    query: str,
    *,
    system: str,
    budget_tokens: int = 800,
    retrieve: bool = True,
    retrieve_k: int = 8,
    retrieve_mode: str = "hybrid",
    memory: Sequence[MemoryItem] = (),
    tools: Sequence[ToolResult] = (),
    ctx: Optional[AssembledContext] = None,
    cache: Optional[Cache] = None,
    cache_config: Optional[CacheConfig] = None,
    router_config: Optional[RouterConfig] = None,
    dry_run: bool = True,
    generate_client=None,
) -> AnswerResult:
    """Answer one query through assemble -> route -> cache -> generate.

    Pass `ctx` when the caller already assembled the context (the CLIs do, so a
    generate run does not pay for retrieval and packing twice).

    dry_run=True never calls a model: a cached answer is still returned (that costs
    nothing), a miss stops at the decision and writes nothing.

    A generation failure is captured in `generate_error` instead of raised, so a
    caller can still report the route, the cache verdict and the assembled prompt;
    the failure is also recorded on the `generate` span whenever the call started.
    """
    memory_items = list(memory or [])
    tool_items = list(tools or [])

    if ctx is None:
        ctx = assemble(AssembleRequest(
            query=query,
            system=system,
            budget_tokens=budget_tokens,
            retrieve=retrieve,
            retrieve_k=retrieve_k,
            retrieve_mode=retrieve_mode,
            memory=memory_items,
            tools=tool_items,
        ))

    decision = route_request(
        query, assemble_meta(ctx, memory_count=len(memory_items)), config=router_config
    )

    fingerprint = fingerprint_for(
        retrieve_mode=retrieve_mode,
        budget_tokens=budget_tokens,
        system=system,
        route=decision.route,
        model=decision.model,
    )

    if cache is None:
        cache_config = cache_config or load_cache_config()
        cache = Cache(cache_config) if cache_config.enabled else None

    if cache is None:
        lookup = CacheLookup(hit=False, score=None, reason="miss", detail="cache disabled")
    else:
        lookup = cache.lookup(query, fingerprint)

    # ── hit: the stored answer is the answer ──────────────────────────────────
    if lookup.hit and lookup.entry is not None:
        return AnswerResult(
            query=query,
            ctx=ctx,
            decision=decision,
            fingerprint=fingerprint,
            lookup=lookup,
            answer=lookup.entry.answer,
            citations=list(lookup.entry.citations),
            skipped_generate=True,
        )

    # ── miss, decision only ───────────────────────────────────────────────────
    if dry_run:
        return AnswerResult(
            query=query,
            ctx=ctx,
            decision=decision,
            fingerprint=fingerprint,
            lookup=lookup,
            skipped_generate=True,
        )

    # ── miss: generate, then remember it ──────────────────────────────────────
    try:
        result = generate(
            GenerateRequest(prompt=ctx.prompt, model=decision.model),
            dry_run=False,
            client=generate_client,
        )
    except Exception as exc:
        return AnswerResult(
            query=query,
            ctx=ctx,
            decision=decision,
            fingerprint=fingerprint,
            lookup=lookup,
            citations=list(ctx.citations),
            generate_error=f"{type(exc).__name__}: {exc}",
        )

    citations = list(ctx.citations)
    wrote: Optional[str] = None
    if cache is not None and result.output:
        entry = cache.write(
            query,
            fingerprint,
            answer=result.output,
            citations=citations,
            model=result.model,
            prompt_tokens=result.input_tokens,
        )
        wrote = entry.cache_id

    return AnswerResult(
        query=query,
        ctx=ctx,
        decision=decision,
        fingerprint=fingerprint,
        lookup=lookup,
        generate=result,
        answer=result.output,
        citations=citations,
        wrote_cache_id=wrote,
    )

"""`python -m contextlab.route` — decide which model should serve a request.

    python -m contextlab.route --query "what does E-4471 mean" --dry-run

Dry-run is the default and the verified path: assemble runs (so n_retrieved and
conflict are real), the policy decides, the cache is consulted, and nothing is
sent to a model. Add --generate to answer with the routed model (needs a real
model name in config/router.yaml plus the OpenAI SDK and key).
"""
import argparse
import sys
from typing import Optional

from contextlab.assemble import assemble, load_system_prompt
from contextlab.cache.answer import answer_with_cache
from contextlab.memory import MemoryStore
from contextlab.trace import start_trace
from contextlab.types import AssembleRequest


def _memory_items(path: Optional[str]) -> list:
    """Memory items from a JSONL file, if one was given."""
    if not path:
        return []
    store = MemoryStore()
    store.load(path)
    return store.items


def _print_decision(decision, prompt_tokens: int) -> None:
    signals = decision.signals
    print("=== Route decision ===")
    print(f"query          {signals.query}")
    print(f"route          {decision.route}")
    print(f"model          {decision.model}")
    print(f"reason         {decision.reason}")
    print(
        "signals        "
        f"intent_hint={signals.intent_hint} has_identifier={str(signals.has_identifier).lower()} "
        f"n_retrieved={signals.n_retrieved} conflict={str(signals.conflict).lower()} "
        f"budget_tokens={signals.budget_tokens}"
    )
    print(f"prompt_tokens  {prompt_tokens}")
    print(f"dry_run        {str(decision.dry_run).lower()}")


def _print_cache(lookup, fingerprint) -> None:
    """Cache verdict — the reason matters more than the boolean."""
    score = f"{lookup.score:.4f}" if lookup.score is not None else "-"
    verdict = "HIT" if lookup.hit else "MISS"
    cache_id = lookup.entry.cache_id if lookup.entry is not None else "-"
    print("\n=== Cache ===")
    print(f"fingerprint    {fingerprint.key()}")
    print(f"result         {verdict}  reason={lookup.reason}  score={score}  cache_id={cache_id}")
    if lookup.detail:
        print(f"detail         {lookup.detail}")


def cmd_route(args: argparse.Namespace) -> int:
    """Assemble, route, consult the cache, and optionally answer."""
    memory = _memory_items(args.memory_file)
    request = AssembleRequest(
        query=args.query,
        system=load_system_prompt(),
        budget_tokens=args.budget,
        retrieve=not args.no_retrieve,
        retrieve_k=args.k,
        retrieve_mode=args.mode,
        memory=memory,
    )

    with start_trace("route", query=args.query, budget_tokens=args.budget) as trace:
        ctx = assemble(request)

        # dry_run unless --generate: a cached answer is still served either way
        answer = answer_with_cache(
            args.query,
            system=request.system,
            budget_tokens=args.budget,
            retrieve=not args.no_retrieve,
            retrieve_k=args.k,
            retrieve_mode=args.mode,
            memory=memory,
            ctx=ctx,
            dry_run=not args.generate,
        )

        _print_decision(answer.decision, ctx.prompt_tokens)
        _print_cache(answer.lookup, answer.fingerprint)

        trace.set_attributes(
            route=answer.decision.route,
            model=answer.decision.model,
            reason=answer.decision.reason,
            intent_hint=answer.decision.signals.intent_hint,
            has_identifier=answer.decision.signals.has_identifier,
            dry_run=answer.decision.dry_run,
            prompt_tokens=ctx.prompt_tokens,
            cache_hit=answer.lookup.hit,
            cache_reason=answer.lookup.reason,
            cache_score=answer.lookup.score,
        )

        if not args.generate:
            print("\n[dry-run] no model was called — pass --generate to answer with the routed model")
        elif answer.lookup.hit:
            print(f"\n=== Answer [cache:{answer.lookup.reason}] ===\n{answer.answer}")
        elif answer.generate_error:
            print(f"\n[generate skipped] {answer.generate_error}", file=sys.stderr)
        else:
            via = (
                f" (fallback from {answer.generate.requested_model})"
                if answer.generate.fallback_used
                else ""
            )
            print(f"\n=== Answer [{answer.generate.model}{via}] ===\n{answer.answer}")

    if trace.trace_id:
        print(
            f"\n[trace] {trace.trace_id} — python -m contextlab.trace show --trace {trace.trace_id}",
            file=sys.stderr,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ContextLab model router (dry-run by default)")
    parser.add_argument("--query", type=str, required=True, help="Query string")
    parser.add_argument("--budget", type=int, default=800, help="Token budget for assembly")
    parser.add_argument("--k", type=int, default=8, help="Retrieval k")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["bm25", "dense", "hybrid"])
    parser.add_argument("--memory-file", type=str, default=None, help="Memory JSONL file")
    parser.add_argument("--no-retrieve", action="store_true", help="Skip retrieval")
    parser.add_argument("--dry-run", action="store_true", help="Decision only (the default)")
    parser.add_argument("--generate", action="store_true", help="Call the routed model (live)")
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    return cmd_route(args)


if __name__ == "__main__":
    sys.exit(main())

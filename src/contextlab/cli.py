"""CLI for ContextLab retrieval and assembly."""
import argparse
import json
import sys
from pathlib import Path

from contextlab.assemble import assemble, BudgetError, load_system_prompt
from contextlab.cache.answer import answer_with_cache
from contextlab.memory import MemoryStore
from contextlab.retrieve import retrieve
from contextlab.trace import start_trace
from contextlab.types import AssembleRequest


def _print_trace_hint(trace_id) -> None:
    """Tell the user how to reopen this run's span tree."""
    if trace_id:
        print(
            f"\n[trace] {trace_id} — python -m contextlab.trace show --trace {trace_id}",
            file=sys.stderr,
        )


def cmd_retrieve(args):
    """Run retrieval and print hits. Traced: root span "retrieve"."""
    with start_trace("retrieve", query=args.query, mode=args.mode, k=args.k) as trace:
        response = retrieve(query=args.query, k=args.k, mode=args.mode)

        print(f"\n=== {args.mode.upper()} Retrieval: \"{args.query}\" ===\n")
        print(f"{'Rank':<5} {'Citation':<30} {'RRF':<8} {'BM25':<8} {'Dense':<8} Snippet")
        print("-" * 100)

        for hit in response.hits:
            snippet = hit.text[:80].replace("\n", " ")
            print(f"{hit.rank:<5} {hit.citation:<30} {hit.scores.get('rrf', 0):.4f}   {hit.scores.get('bm25', 0):.4f}   {hit.scores.get('dense', 0):.4f}   {snippet}")

        trace.set_attributes(
            n_hits=len(response.hits),
            chunk_ids=[hit.chunk_id for hit in response.hits],
            doc_ids=sorted({hit.doc_id for hit in response.hits}),
        )

    _print_trace_hint(trace.trace_id)


def cmd_assemble(args):
    """Run context assembly and print prompt. Traced: root span "assemble"."""
    # Load system prompt
    if args.system_file:
        with open(args.system_file) as f:
            system_text = f.read()
    else:
        system_text = load_system_prompt()

    # Load memory
    memory_items = []
    if args.memory_file:
        store = MemoryStore()
        store.load(args.memory_file)
        memory_items = store.items

    # Build request
    request = AssembleRequest(
        query=args.query,
        system=system_text,
        budget_tokens=args.budget,
        retrieve=not args.no_retrieve,
        retrieve_k=args.k,
        retrieve_mode=args.mode,
        memory=memory_items,
        tools=[],
    )

    with start_trace(
        "assemble", query=args.query, budget_tokens=args.budget, k=args.k, mode=args.mode
    ) as trace:
        try:
            ctx = assemble(request)
        except BudgetError as e:
            trace.set_attributes(budget_error=str(e))
            trace.root.fail(e)
            print(f"BudgetError: {e}", file=sys.stderr)
            sys.exit(1)

        # Print output
        print(f"=== Budget: {ctx.budget_tokens} | Prompt tokens: {ctx.prompt_tokens} | Dropped: {len(ctx.dropped)} ===\n")

        if ctx.citations:
            print("=== Citations (kept) ===")
            for c in ctx.citations:
                print(f"  {c}")
            print()

        print("=== Prompt ===")
        print(ctx.prompt)
        print()

        if ctx.dropped:
            print("=== Dropped ===")
            for b in ctx.dropped:
                print(f"  [{b.kind}] {b.ref_id} ({b.token_count} tokens)")

        trace.set_attributes(
            prompt_tokens=ctx.prompt_tokens,
            kept_ids=[b.ref_id for b in ctx.blocks if b.kind in ("retrieval", "memory", "tool")],
            dropped_ids=[b.ref_id for b in ctx.dropped],
        )

        # Optional generate: router picks the model, cache may already have the answer.
        if args.generate:
            try:
                answer = answer_with_cache(
                    args.query,
                    system=system_text,
                    budget_tokens=args.budget,
                    retrieve=not args.no_retrieve,
                    retrieve_k=args.k,
                    retrieve_mode=args.mode,
                    memory=memory_items,
                    ctx=ctx,
                    dry_run=False,
                )
            except Exception as exc:  # cache/encoder failure: report, do not crash
                print(f"\n[generate failed] {type(exc).__name__}: {exc}", file=sys.stderr)
            else:
                print(
                    f"\n[router] route={answer.decision.route} model={answer.decision.model} "
                    f"({answer.decision.reason})",
                    file=sys.stderr,
                )
                if answer.lookup.hit:
                    print(
                        f"[cache] HIT ({answer.lookup.reason}) score={answer.lookup.score} "
                        f"id={answer.lookup.entry.cache_id}",
                        file=sys.stderr,
                    )
                    print(f"\n=== Answer [cache:{answer.lookup.reason}] ===\n{answer.answer}")
                else:
                    print(f"[cache] miss ({answer.lookup.reason})", file=sys.stderr)
                    if answer.generate_error:
                        print(f"[generate skipped] {answer.generate_error}", file=sys.stderr)
                    else:
                        via = (
                            f" fallback from {answer.generate.requested_model}"
                            if answer.generate.fallback_used
                            else ""
                        )
                        print(f"\n=== Answer [{answer.generate.model}{via}] ===\n{answer.answer}")

    _print_trace_hint(trace.trace_id)


def main():
    parser = argparse.ArgumentParser(description="ContextLab CLI")
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # retrieve subcommand
    r = sub.add_parser("retrieve", help="Run retrieval")
    r.add_argument("--query", type=str, required=True, help="Query string")
    r.add_argument("--k", type=int, default=5, help="Number of hits")
    r.add_argument("--mode", type=str, default="hybrid", choices=["bm25", "dense", "hybrid"], help="Retrieval mode")

    # assemble subcommand
    a = sub.add_parser("assemble", help="Assemble a context")
    a.add_argument("--query", type=str, required=True, help="Query string")
    a.add_argument("--budget", type=int, default=800, help="Token budget")
    a.add_argument("--k", type=int, default=8, help="Retrieval k")
    a.add_argument("--mode", type=str, default="hybrid", choices=["bm25", "dense", "hybrid"], help="Retrieval mode")
    a.add_argument("--memory-file", type=str, default=None, help="Memory JSONL file")
    a.add_argument("--system-file", type=str, default=None, help="System prompt file")
    a.add_argument("--generate", action="store_true", help="Generate response via API")
    a.add_argument("--no-retrieve", action="store_true", help="Skip retrieval")

    args = parser.parse_args()

    if args.command == "retrieve":
        cmd_retrieve(args)
    elif args.command == "assemble":
        cmd_assemble(args)
    else:
        # Backward compat: if no subcommand, treat as old retrieve syntax
        if "--query" in sys.argv:
            # Re-parse with retrieve-style args
            parser1 = argparse.ArgumentParser(description="ContextLab retrieval")
            parser1.add_argument("--query", type=str, required=True)
            parser1.add_argument("--k", type=int, default=5)
            parser1.add_argument("--mode", type=str, default="hybrid", choices=["bm25", "dense", "hybrid"])
            args1 = parser1.parse_args()
            cmd_retrieve(args1)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()

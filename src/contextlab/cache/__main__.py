"""`python -m contextlab.cache` — inspect, seed and measure the semantic cache.

    python -m contextlab.cache lookup --query "What does E-4471 mean?"
    python -m contextlab.cache seed --from-evals
    python -m contextlab.cache stats
    python -m contextlab.cache clear --yes
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from contextlab.assemble import load_system_prompt
from contextlab.cache.answer import answer_with_cache
from contextlab.cache.config import CacheConfig, load_cache_config
from contextlab.cache.fingerprint import fingerprint_from_dict
from contextlab.cache.lookup import Cache
from contextlab.cache.store import CacheStore
from contextlab.memory import MemoryStore
from contextlab.router import load_router_config, route_request
from contextlab.trace import start_trace

DEFAULT_GOLDEN = "evals/cache_golden.jsonl"


def _memory_items(path: Optional[str]) -> list:
    """Memory items from a JSONL file, if one was given."""
    if not path:
        return []
    store = MemoryStore()
    store.load(path)
    return store.items


def _print_lookup(result, config: CacheConfig) -> None:
    lookup = result.lookup
    print("=== Cache lookup ===")
    print(f"query          {result.query}")
    print(f"fingerprint    {result.fingerprint.key()}")
    print(f"threshold      {config.threshold}  (require_fingerprint={str(config.require_fingerprint).lower()})")
    verdict = "HIT" if lookup.hit else "MISS"
    score = f"{lookup.score:.4f}" if lookup.score is not None else "-"
    cache_id = lookup.entry.cache_id if lookup.entry is not None else "-"
    print(f"result         {verdict}  reason={lookup.reason}  score={score}  cache_id={cache_id}")
    if lookup.detail:
        print(f"detail         {lookup.detail}")
    if lookup.entry is not None:
        print(f"cached_query   {lookup.entry.query}")
        print(f"model          {lookup.entry.model or '-'}   prompt_tokens={lookup.entry.prompt_tokens or '-'}")
        print(f"citations      {', '.join(lookup.entry.citations) or '-'}")
        print(f"answer         {lookup.entry.answer}")


def cmd_lookup(args: argparse.Namespace) -> int:
    """Look up one query exactly the way the answer path does (no model call)."""
    config = load_cache_config(args.config)
    memory = _memory_items(args.memory_file)

    with start_trace("cache", query=args.query, budget_tokens=args.budget) as trace:
        result = answer_with_cache(
            args.query,
            system=load_system_prompt(),
            budget_tokens=args.budget,
            retrieve=not args.no_retrieve,
            retrieve_k=args.k,
            retrieve_mode=args.mode,
            memory=memory,
            cache_config=config,
            dry_run=True,
        )

        _print_lookup(result, config)
        print(f"\n[route] {result.decision.route} / {result.decision.model} ({result.decision.reason})")

        trace.set_attributes(
            hit=result.lookup.hit,
            score=result.lookup.score,
            reason=result.lookup.reason,
            cache_id=result.lookup.entry.cache_id if result.lookup.entry else None,
            store_path=str(config.resolved_path()),
        )

    if trace.trace_id:
        print(
            f"\n[trace] {trace.trace_id} — python -m contextlab.trace show --trace {trace.trace_id}",
            file=sys.stderr,
        )
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    """Seed the cache from evals/cache_golden.jsonl (no LLM involved)."""
    golden = Path(args.golden)
    if not golden.exists():
        print(f"golden file not found: {golden}", file=sys.stderr)
        return 1

    config = load_cache_config(args.config)
    if not config.enabled:
        print(f"cache disabled in {config.source}", file=sys.stderr)
        return 1

    router_config = load_router_config()
    cache = Cache(config, CacheStore(config.resolved_path()))
    written = 0

    with start_trace("cache.seed", golden=str(golden)) as trace:
        for line in golden.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            # Seeded entries fingerprint like real ones: include the route and model
            # the router would pick, or the answer path would never match them.
            decision = route_request(row["seed_query"], None, config=router_config)
            fingerprint = fingerprint_from_dict(
                {**(row.get("fingerprint") or {}), "route": decision.route, "model": decision.model}
            )
            entry = cache.write(
                row["seed_query"],
                fingerprint,
                answer=row["seed_answer"],
                citations=row.get("seed_citations") or [],
                model=decision.model,
            )
            written += 1
            print(f"  seeded {row['id']:6} {entry.cache_id}  {row['seed_query']}")

        trace.set_attributes(entries=written, store_path=str(config.resolved_path()))

    print(f"\n{written} entries -> {config.resolved_path()}")
    print("note: seeded with query-only route signals; the answer path routes after assemble")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Show what the store holds."""
    config = load_cache_config(args.config)
    stats = CacheStore(config.resolved_path()).stats()
    print(f"path           {config.resolved_path()}")
    print(f"threshold      {config.threshold}   require_fingerprint={config.require_fingerprint}")
    for key, value in stats.items():
        print(f"{key:15}{value}")
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    """Delete every entry (needs --yes)."""
    config = load_cache_config(args.config)
    if not args.yes:
        print(f"refusing to delete {config.resolved_path()} without --yes", file=sys.stderr)
        return 1
    CacheStore(config.resolved_path()).clear()
    print(f"cleared {config.resolved_path()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m contextlab.cache", description="Semantic cache tools")
    sub = parser.add_subparsers(dest="command")

    lookup = sub.add_parser("lookup", help="look up one query (no model call)")
    lookup.add_argument("--query", type=str, required=True)
    lookup.add_argument("--budget", type=int, default=800)
    lookup.add_argument("--k", type=int, default=8)
    lookup.add_argument("--mode", type=str, default="hybrid", choices=["bm25", "dense", "hybrid"])
    lookup.add_argument("--memory-file", type=str, default=None)
    lookup.add_argument("--no-retrieve", action="store_true")
    lookup.add_argument("--config", type=str, default=None, help="cache config path override")
    lookup.set_defaults(func=cmd_lookup)

    seed = sub.add_parser("seed", help="seed the cache from the golden file")
    seed.add_argument("--from-evals", action="store_true", help="seed from evals/cache_golden.jsonl")
    seed.add_argument("--golden", type=str, default=DEFAULT_GOLDEN)
    seed.add_argument("--config", type=str, default=None)
    seed.set_defaults(func=cmd_seed)

    stats = sub.add_parser("stats", help="show store contents summary")
    stats.add_argument("--config", type=str, default=None)
    stats.set_defaults(func=cmd_stats)

    clear = sub.add_parser("clear", help="delete every cached entry")
    clear.add_argument("--yes", action="store_true", help="confirm deletion")
    clear.add_argument("--config", type=str, default=None)
    clear.set_defaults(func=cmd_clear)

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    if getattr(args, "from_evals", False) is False and args.command == "seed":
        print("seed needs --from-evals", file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

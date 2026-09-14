"""Entry point: python -m contextlab.trace show [--last|--trace ID|--case ID]"""
import argparse
import sys
from typing import Optional, Sequence

from contextlab.trace import show as show_module
from contextlab.trace.export import load_records, trace_dir


def _resolve(args: argparse.Namespace):
    """Load records and pick the trace the user asked for."""
    records = load_records(args.dir)
    if not records:
        print(
            f"no spans in {trace_dir(args.dir)} — run a traced command first "
            f"(python -m contextlab assemble --query \"...\" --budget 800)",
            file=sys.stderr,
        )
        return None, None

    last = args.last or not (args.trace or args.case)
    trace_id = show_module.select_trace_id(
        records, trace_id=args.trace, case_id=args.case, last=last
    )
    if trace_id is None:
        what = f"trace {args.trace}" if args.trace else f"case {args.case}"
        print(f"no trace found for {what} ({len(records)} spans scanned)", file=sys.stderr)
        return None, None
    return records, trace_id


def cmd_show(args: argparse.Namespace) -> int:
    """Print the summary and span tree of one trace."""
    records, trace_id = _resolve(args)
    if records is None:
        return 1

    spans = show_module.spans_of(records, trace_id)
    print(show_module.render_trace(spans, full=args.full))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List recent traces, newest first."""
    records = load_records(args.dir)
    print(f"dir {trace_dir(args.dir)}")
    print(show_module.render_trace_list(records, limit=args.limit))
    return 0 if records else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m contextlab.trace", description="Inspect ContextLab traces"
    )
    sub = parser.add_subparsers(dest="command")

    show = sub.add_parser("show", help="print a trace's span tree")
    selector = show.add_mutually_exclusive_group()
    selector.add_argument("--last", action="store_true", help="most recent trace (default)")
    selector.add_argument("--trace", type=str, help="trace id (32 hex chars)")
    selector.add_argument("--case", type=str, help="latest trace containing this eval case id")
    show.add_argument("--dir", type=str, default=None, help="trace directory (default: config/trace.yaml)")
    show.add_argument("--full", action="store_true", help="do not truncate attribute values")
    show.set_defaults(func=cmd_show)

    listing = sub.add_parser("list", help="list recent traces")
    listing.add_argument("--dir", type=str, default=None, help="trace directory")
    listing.add_argument("--limit", type=int, default=20, help="how many traces to show")
    listing.set_defaults(func=cmd_list)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

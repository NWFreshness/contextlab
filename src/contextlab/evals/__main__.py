"""Entry point: python -m contextlab.evals"""
import argparse
import sys

from contextlab.evals.runner import cmd_run


def main() -> int:
    parser = argparse.ArgumentParser(prog="contextlab.evals")
    sub = parser.add_subparsers(dest="command", required=True)

    # run subcommand
    run_p = sub.add_parser("run", help="Run eval suite(s)")
    run_p.add_argument("--suite", default="all", choices=["retrieval", "assembly", "answer", "router", "cache", "all"])
    run_p.add_argument("--offline", action="store_true", help="Skip LLM generation in answer suite")
    run_p.add_argument("--require-answer", action="store_true", help="Fail if answer suite cannot run")
    run_p.set_defaults(func=cmd_run)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

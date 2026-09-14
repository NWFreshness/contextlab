"""Entry point for `python -m contextlab` with subcommand routing."""
import argparse
import sys


def _parse_assemble_args():
    """Parse assemble subcommand args (stripping 'assemble' from sys.argv)."""
    # sys.argv[0] = module path, sys.argv[1] = 'assemble'
    # We need to pass only the args AFTER 'assemble'
    argv = sys.argv[2:] if len(sys.argv) > 2 else []
    parser = argparse.ArgumentParser(description="ContextLab assembler")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--mode", type=str, default="hybrid", choices=["bm25", "dense", "hybrid"])
    parser.add_argument("--memory-file", type=str, default=None)
    parser.add_argument("--system-file", type=str, default=None)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--no-retrieve", action="store_true")
    return parser.parse_args(argv)


def _parse_retrieve_args():
    """Parse retrieve subcommand args (stripping 'retrieve' from sys.argv)."""
    argv = sys.argv[2:] if len(sys.argv) > 2 else []
    parser = argparse.ArgumentParser(description="ContextLab retrieval")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", type=str, default="hybrid", choices=["bm25", "dense", "hybrid"])
    return parser.parse_args(argv)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "assemble":
        from contextlab.cli import cmd_assemble
        args = _parse_assemble_args()
        cmd_assemble(args)
    elif len(sys.argv) > 1 and sys.argv[1] == "retrieve":
        from contextlab.cli import cmd_retrieve
        args = _parse_retrieve_args()
        cmd_retrieve(args)
    else:
        from contextlab.cli import main
        main()

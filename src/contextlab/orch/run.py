"""CLI entry point: `python -m contextlab.orch run ...`.

Mirrors the existing CLI shape (`python -m contextlab.route --query …`) so
the project's command surface stays consistent.

Examples:
    python -m contextlab.orch run --query "what does E-4471 mean" --driver script
    python -m contextlab.orch run --query "..." --driver script --max-steps 6
    python -m contextlab.orch run --query "..." --driver llm  # skip-safe stub
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from contextlab.orch.machine import orchestrate as orch_run
from contextlab.orch.state import (
    OrchRequest,
    load_orchestrator_config,
    new_trajectory_id,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ContextLab bounded agent orchestrator"
    )
    sub = parser.add_subparsers(dest="orch_command", required=True)

    p = sub.add_parser("run", help="Run the orchestrator")
    p.add_argument("--query", type=str, required=True, help="User query")
    p.add_argument("--budget", type=int, default=800, help="Token budget for assembly")
    p.add_argument("--k", type=int, default=5, help="Retrieval k (top hits per query)")
    p.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["bm25", "dense", "hybrid"],
        help="Retrieval mode for the orchestrator's `retrieve` tool",
    )
    p.add_argument(
        "--driver",
        type=str,
        default="script",
        choices=["script", "llm"],
        help="Policy driver. `script` is the verified default; `llm` is a stub.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max_steps from config (default: from config/orchestrator.yaml)",
    )
    p.add_argument(
        "--trajectory-dir",
        type=str,
        default=None,
        help="Directory for the trajectory file (default: artifacts/trajectories/)",
    )
    p.add_argument(
        "--system-file",
        type=str,
        default=None,
        help="Override system prompt file",
    )
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    """Resolve config, build request, run the machine, print the result."""
    config = load_orchestrator_config()
    max_steps = args.max_steps if args.max_steps is not None else int(config.get("max_steps", 6))
    tools = list(config.get("tools", ["retrieve", "read_chunk", "finish"]))

    system_text = None
    if args.system_file:
        with open(args.system_file) as f:
            system_text = f.read()

    traj_dir = (
        Path(args.trajectory_dir)
        if args.trajectory_dir
        else PROJECT_ROOT / "artifacts" / "trajectories"
    )

    request = OrchRequest(
        query=args.query,
        budget_tokens=args.budget,
        retrieve_k=args.k,
        retrieve_mode=args.mode,
        system=system_text,
        driver=args.driver,
        max_steps=max_steps,
        tools=tools,
        trajectory_dir=traj_dir,
    )

    trajectory = orch_run(request)

    # ── Pretty output ────────────────────────────────────────────────────
    print(f"=== Orchestrator run ({trajectory.driver}) ===")
    print(f"query           {trajectory.query}")
    print(f"trajectory_id   {trajectory.trajectory_id}")
    print(f"stop_reason     {trajectory.stop_reason}")
    print(f"n_steps         {len(trajectory.steps)}")
    if trajectory.citations:
        print(f"citations       {', '.join(trajectory.citations)}")
    if trajectory.prompt_tokens is not None:
        print(f"prompt_tokens   {trajectory.prompt_tokens}")
    if trajectory.trace_id:
        print(
            f"[trace] {trajectory.trace_id} — "
            f"python -m contextlab.trace show --trace {trajectory.trace_id}"
        )

    print("\n=== Steps ===")
    for s in trajectory.steps:
        kind = s.action.type
        tool = s.action.tool or "-"
        reason = s.action.reason or "-"
        # observation preview
        obs = s.observation or ""
        preview = obs[:60].replace("\n", " ")
        print(
            f"  step={s.index:>2} state={s.state_name:<8} type={kind:<10} "
            f"tool={tool:<12} reason={reason:<24} obs={preview!r}"
        )

    if trajectory.final_answer:
        print(f"\n=== Final answer ===\n{trajectory.final_answer[:400]}")

    # ── Exit code ────────────────────────────────────────────────────────
    if trajectory.stop_reason == "done":
        return 0
    return 2  # any non-done stop is a non-success run


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.orch_command == "run":
        return cmd_run(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
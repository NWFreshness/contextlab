"""Eval runner — orchestrates suites, applies gates, writes report."""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from contextlab.evals.types import CaseResult, CheckResult, EvalReport, SuiteMetrics
from contextlab.evals.report import format_summary, write_latest, write_timestamped
from contextlab.evals import suites_retrieval, suites_assembly, suites_answer, suites_router, suites_cache
from contextlab.config import get_config


def load_gates(gates_path: str | Path = "evals/gates.yaml") -> dict:
    """Load gates from YAML."""
    p = Path(gates_path)
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def apply_gates(suites_results: dict[str, tuple[list[CaseResult], SuiteMetrics]], gates: dict) -> list[CheckResult]:
    """Apply gate thresholds to suite results. Returns list of gate CheckResults."""
    gate_results: list[CheckResult] = []

    for suite_name, gate_config in gates.items():
        if suite_name not in suites_results:
            continue
        results, metrics = suites_results[suite_name]

        if suite_name == "retrieval":
            # Gate on recall@5 thresholds defined in gates.yaml
            intent_map = {
                "hybrid": "min_recall_at_5_hybrid",
                "lexical": "min_recall_at_5_lexical",
                "paraphrase": "min_recall_at_5_paraphrase",
            }
            for intent_key, gate_key in intent_map.items():
                threshold = gate_config.get(gate_key)
                if threshold is None:
                    continue
                metric_key = f"recall@5_{intent_key}"
                actual = metrics.metrics.get(metric_key, 0.0)
                passed = actual >= threshold
                gate_results.append(CheckResult(
                    name=f"gate_retrieval_{intent_key}",
                    passed=passed,
                    detail=f"{metric_key}={actual:.3f} >= {threshold} => {'PASS' if passed else 'FAIL'}",
                    score=actual,
                ))

        elif suite_name == "assembly":
            violations = sum(1 for r in results if any(
                c.name == "under_budget" and not c.passed for c in r.checks
            ))
            max_violations = gate_config.get("max_budget_violations", 0)
            citation_errors = sum(1 for r in results if any(
                c.name == "citations_resolve" and not c.passed for c in r.checks
            ))
            max_citation_errors = gate_config.get("max_citation_errors", 0)

            gate_results.append(CheckResult(
                name="gate_assembly_budget",
                passed=violations <= max_violations,
                detail=f"budget violations={violations} <= max={max_violations}",
                score=1.0 if violations <= max_violations else 0.0,
            ))
            gate_results.append(CheckResult(
                name="gate_assembly_citations",
                passed=citation_errors <= max_citation_errors,
                detail=f"citation errors={citation_errors} <= max={max_citation_errors}",
                score=1.0 if citation_errors <= max_citation_errors else 0.0,
            ))

        elif suite_name == "answer":
            if gate_config.get("min_pass_rate") is not None:
                n = len(results)
                n_pass = sum(1 for r in results if r.passed)
                rate = n_pass / n if n else 0.0
                threshold = gate_config["min_pass_rate"]
                gate_results.append(CheckResult(
                    name="gate_answer_pass_rate",
                    passed=rate >= threshold,
                    detail=f"pass_rate={rate:.2f} >= {threshold}",
                    score=rate,
                ))
            max_forbidden = gate_config.get("max_forbidden_fact_hits", 0)
            forbidden_hits = sum(
                sum(1 for c in r.checks if c.name == "forbidden_facts" and not c.passed)
                for r in results
            )
            gate_results.append(CheckResult(
                name="gate_answer_forbidden",
                passed=forbidden_hits <= max_forbidden,
                detail=f"forbidden_fact_hits={forbidden_hits} <= max={max_forbidden}",
                score=1.0 if forbidden_hits <= max_forbidden else 0.0,
            ))

        elif suite_name == "router":
            if gate_config.get("min_accuracy") is not None:
                threshold = gate_config["min_accuracy"]
                actual = metrics.metrics.get("route_accuracy", 0.0)
                gate_results.append(CheckResult(
                    name="gate_router_accuracy",
                    passed=actual >= threshold,
                    detail=f"route_accuracy={actual:.3f} >= {threshold}",
                    score=actual,
                ))

        elif suite_name == "cache":
            min_hit = gate_config.get("min_should_hit_rate")
            if min_hit is not None:
                actual = metrics.metrics.get("should_hit_rate", 0.0)
                gate_results.append(CheckResult(
                    name="gate_cache_should_hit",
                    passed=actual >= min_hit,
                    detail=f"should_hit_rate={actual:.3f} >= {min_hit}",
                    score=actual,
                ))
            max_false = gate_config.get("max_false_hit_rate")
            if max_false is not None:
                actual = metrics.metrics.get("false_hit_rate", 0.0)
                gate_results.append(CheckResult(
                    name="gate_cache_false_hit",
                    passed=actual <= max_false,
                    detail=f"false_hit_rate={actual:.3f} <= {max_false}",
                    score=actual,
                ))

    return gate_results


def run_suites(suite_names: list[str], offline: bool, require_answer: bool) -> dict[str, tuple[list[CaseResult], SuiteMetrics]]:
    """Run selected suites. Returns {name: (results, metrics)}."""
    results: dict[str, tuple[list[CaseResult], SuiteMetrics]] = {}

    for name in suite_names:
        if name == "retrieval":
            print("Running retrieval suite...", flush=True)
            results["retrieval"] = suites_retrieval.run_suite()
        elif name == "assembly":
            print("Running assembly suite...", flush=True)
            results["assembly"] = suites_assembly.run_suite()
        elif name == "answer":
            print("Running answer suite...", flush=True)
            results["answer"] = suites_answer.run_suite(offline=offline)
        elif name == "router":
            # The router suite is always dry-run: it grades decisions, never calls a model.
            print("Running router suite (dry-run)...", flush=True)
            results["router"] = suites_router.run_suite()
        elif name == "cache":
            # The cache suite is offline by construction: golden fixtures, no LLM.
            print("Running cache suite (seeded from goldens)...", flush=True)
            results["cache"] = suites_cache.run_suite()
        elif name == "all":
            print("Running retrieval suite...", flush=True)
            results["retrieval"] = suites_retrieval.run_suite()
            print("Running assembly suite...", flush=True)
            results["assembly"] = suites_assembly.run_suite()
            print("Running answer suite...", flush=True)
            results["answer"] = suites_answer.run_suite(offline=offline)
            print("Running router suite (dry-run)...", flush=True)
            results["router"] = suites_router.run_suite()
            print("Running cache suite (seeded from goldens)...", flush=True)
            results["cache"] = suites_cache.run_suite()

    return results


def cmd_run(args: argparse.Namespace) -> int:
    """Main eval runner command. Returns exit code."""
    suites_to_run = (
        ["retrieval", "assembly", "answer", "router", "cache"] if args.suite == "all" else [args.suite]
    )
    offline = args.offline
    require_answer = args.require_answer

    config = get_config()
    started_at = datetime.now(timezone.utc).isoformat()

    print(f"Starting eval run at {started_at}", flush=True)
    print(f"Suites: {suites_to_run}, offline={offline}, require_answer={require_answer}", flush=True)

    all_results: dict[str, tuple[list[CaseResult], SuiteMetrics]] = {}

    # Check answer suite availability
    has_answer_key = bool(__import__("os").environ.get("OPENAI_API_KEY"))

    if "answer" in suites_to_run and offline:
        print("Running in OFFLINE mode — answer generation will be skipped", flush=True)
    elif "answer" in suites_to_run and not has_answer_key:
        print("Warning: OPENAI_API_KEY not set — answer generation will be skipped", flush=True)

    # Run suites
    suite_results = run_suites(suites_to_run, offline, require_answer)
    all_results.update(suite_results)

    # Load gates
    gates = load_gates()

    # Apply gates
    gate_results = apply_gates(all_results, gates)

    # Build report
    finished_at = datetime.now(timezone.utc).isoformat()

    report = EvalReport(
        started_at=started_at,
        finished_at=finished_at,
        settings={
            "config": dict(config),
            "offline": offline,
            "require_answer": require_answer,
            "suites_run": suites_to_run,
        },
        suites={name: metrics for name, (_, metrics) in all_results.items()},
        gates=gate_results,
        cases=[r for _, (results, _) in all_results.items() for r in results],
        passed=all(g.passed for g in gate_results),
    )

    # Write reports
    write_latest(report)
    write_timestamped(report)

    # Print summary
    print("\n" + format_summary(report), flush=True)

    # Exit code
    if report.passed:
        print("\n=== EVAL PASSED ===", flush=True)
        return 0
    else:
        print("\n=== EVAL FAILED ===", flush=True)
        return 1


def add_run_command(subparsers) -> None:
    """Add the run subcommand to an argparse parser."""
    p = subparsers.add_parser("run", help="Run eval suite(s)")
    p.add_argument("--suite", default="all", choices=["retrieval", "assembly", "answer", "router", "cache", "all"])
    p.add_argument("--offline", action="store_true", help="Skip LLM generation in answer suite")
    p.add_argument("--require-answer", action="store_true", help="Fail if answer suite cannot run")
    p.set_defaults(func=cmd_run)

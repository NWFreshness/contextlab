"""Router suite — dry-run route accuracy against evals/router_golden.jsonl.

No HTTP: every case is assembled (so n_retrieved / conflict are real), the policy
decides, and the decision is graded against the case's expected_route. Each case
gets its own `eval.case` trace holding the `router.decide` span.

Two metrics, one gate: the gate is on route_accuracy (the brief's contract); the
case pass/fail also requires the golden's `intent` label to match the policy's
intent_hint, so a keyword edit that silently re-labels cases fails loudly instead
of moving the number.
"""
import json
import time
from pathlib import Path
from typing import Optional

from contextlab.assemble import assemble
from contextlab.evals.types import CaseResult, CheckResult, EvalCase, SuiteMetrics
from contextlab.router import (
    RouterConfig,
    assemble_meta,
    load_router_config,
    route_request,
)
from contextlab.trace import start_trace
from contextlab.types import AssembleRequest, MemoryItem, ToolResult

SYSTEM = "You are a helpful assistant."


class RouterCase(EvalCase):
    """Golden case plus the two fields only the router suite needs.

    A subclass on purpose: Slice 3's EvalCase stays untouched, so the assembly
    suite's `getattr(case, "retrieve", False)` keeps behaving exactly as before.
    """

    expected_route: str = ""
    retrieve: bool = False


def load_golden(path: str | Path = "evals/router_golden.jsonl") -> list[RouterCase]:
    """Load router golden cases."""
    cases: list[RouterCase] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            data["suite"] = "router"  # override suite from file
            cases.append(RouterCase(**data))
    return cases


def run_suite(
    golden_path: str | Path = "evals/router_golden.jsonl",
    config: Optional[RouterConfig] = None,
) -> tuple[list[CaseResult], SuiteMetrics]:
    """Route every golden case in dry-run mode and grade the decision."""
    config = config or load_router_config()
    cases = load_golden(golden_path)
    results: list[CaseResult] = []

    n_routed = 0
    n_intent = 0
    route_counts: dict[str, int] = {}

    for case in cases:
        if case.expected_route not in config.models:
            raise ValueError(
                f"{case.id}: expected_route '{case.expected_route}' is not in models "
                f"({sorted(config.models)}) — fix the golden, not the gate"
            )

        with start_trace("eval.case", case_id=case.id, suite="router", query=case.query) as trace:
            t0 = time.perf_counter()
            memory = [MemoryItem(**m) for m in case.memory_items]
            tools = [ToolResult(**t) for t in case.tool_results]

            request = AssembleRequest(
                query=case.query,
                system=SYSTEM,
                budget_tokens=case.budget_tokens or 800,
                retrieve=case.retrieve,
                retrieve_k=8,
                tools=tools,
                memory=memory,
            )
            ctx = assemble(request)

            decision = route_request(
                case.query,
                assemble_meta(ctx, memory_count=len(memory)),
                config=config,
            )

            route_ok = decision.route == case.expected_route
            intent_ok = decision.signals.intent_hint == case.intent
            n_routed += int(route_ok)
            n_intent += int(intent_ok)
            route_counts[decision.route] = route_counts.get(decision.route, 0) + 1

            checks = [
                CheckResult(
                    name="route_match",
                    passed=route_ok,
                    detail=f"expected={case.expected_route} got={decision.route} ({decision.reason})",
                    score=1.0 if route_ok else 0.0,
                ),
                CheckResult(
                    name="intent_label",
                    passed=intent_ok,
                    detail=f"label={case.intent or '-'} signals={decision.signals.intent_hint}",
                    score=1.0 if intent_ok else 0.0,
                ),
            ]
            passed = route_ok and intent_ok

            trace.set_attributes(
                passed=passed,
                expected_route=case.expected_route,
                route=decision.route,
                model=decision.model,
                intent_hint=decision.signals.intent_hint,
                conflict=decision.signals.conflict,
                # hits the retriever actually returned (0 when the case skipped retrieval)
                n_retrieved=decision.signals.n_retrieved,
            )

            results.append(CaseResult(
                id=case.id,
                suite="router",
                passed=passed,
                checks=checks,
                cited_ids=[b.ref_id for b in ctx.blocks if b.kind in ("retrieval", "memory")],
                prompt_tokens=ctx.prompt_tokens,
                latency_ms=(time.perf_counter() - t0) * 1000,
                trace_id=trace.trace_id,
                route=decision.route,
                model=decision.model,
            ))

    n = len(cases)
    metrics = {
        "route_accuracy": n_routed / n if n else 0.0,
        "intent_accuracy": n_intent / n if n else 0.0,
        "routes_cheap": route_counts.get("cheap", 0),
        "routes_strong": route_counts.get("strong", 0),
        "routes_fallback": route_counts.get("fallback", 0),
        "cases_with_retrieval": sum(1 for c in cases if c.retrieve),
    }

    sm = SuiteMetrics(
        n=n,
        n_pass=sum(1 for r in results if r.passed),
        n_fail=sum(1 for r in results if not r.passed),
        metrics=metrics,
    )
    return results, sm

"""Assembly suite — deterministic invariant checks.

Each case runs inside its own root trace named "eval.case" holding the
assemble.pack span (kept/dropped ids) and any retrieve.* hops underneath.
"""
import json
import time
from pathlib import Path

from contextlab.evals.types import CaseResult, CheckResult, EvalCase, SuiteMetrics
from contextlab.assemble import assemble, BudgetError
from contextlab.tokenize import count_tokens
from contextlab.trace import start_trace


def load_golden(path: str | Path = "evals/assembly_golden.jsonl") -> list[EvalCase]:
    """Load assembly golden cases."""
    cases = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            d["suite"] = "assembly"  # override suite from file
            cases.append(EvalCase(**d))
    return cases


def run_suite(golden_path: str | Path = "evals/assembly_golden.jsonl") -> tuple[list[CaseResult], SuiteMetrics]:
    """Run assembly suite — deterministic invariant checks."""
    cases = load_golden(golden_path)
    results: list[CaseResult] = []

    # Load valid chunk ids
    chunk_ids: set[str] = set()
    chunk_path = Path("data/chunks.jsonl")
    if chunk_path.exists():
        with open(chunk_path) as f:
            for line in f:
                d = json.loads(line)
                chunk_ids.add(d["chunk_id"])

    for case in cases:
        with start_trace("eval.case", case_id=case.id, suite="assembly", query=case.query) as trace:
            t0 = time.perf_counter()
            checks: list[CheckResult] = []

            from contextlab.types import AssembleRequest, MemoryItem, ToolResult

            memory = [MemoryItem(**m) for m in case.memory_items]
            tools = [ToolResult(**t) for t in case.tool_results]

            try:
                request = AssembleRequest(
                    query=case.query,
                    system="You are a helpful assistant.",
                    budget_tokens=case.budget_tokens or 2000,
                    retrieve=getattr(case, "retrieve", False),
                    memory=memory,
                    tools=tools,
                )
                ctx = assemble(request)
                prompt_tokens = ctx.prompt_tokens
                prompt = ctx.prompt
                cited_ids = [b.ref_id for b in ctx.blocks if b.kind in ("retrieval", "memory")]
                dropped_ids = [b.ref_id for b in ctx.dropped]
            except BudgetError as e:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                trace.set_attributes(passed=False, budget_error=str(e))
                results.append(CaseResult(
                    id=case.id,
                    suite="assembly",
                    passed=False,
                    checks=[CheckResult(name="no_budget_error", passed=False, detail=str(e))],
                    prompt_tokens=None,
                    latency_ms=elapsed_ms,
                    trace_id=trace.trace_id,
                ))
                continue

            # ── Invariant checks ────────────────────────────────────────────────

            # under_budget: prompt_tokens <= budget
            checks.append(CheckResult(
                name="under_budget",
                passed=prompt_tokens <= (case.budget_tokens or 2000),
                detail=f"prompt_tokens={prompt_tokens} <= budget={case.budget_tokens}",
                score=1.0 if prompt_tokens <= (case.budget_tokens or 2000) else 0.0,
            ))

            # query_kept: query text present in prompt
            checks.append(CheckResult(
                name="query_kept",
                passed=case.query in prompt,
                detail="query present" if case.query in prompt else "query MISSING from prompt",
            ))

            # system_kept: system prompt text present
            checks.append(CheckResult(
                name="system_kept",
                passed="You are a helpful assistant." in prompt,
                detail="system prompt present",
            ))

            # citations_resolve: every cited id exists in chunks or is a known tool/memory id
            cited_refs = set(cited_ids)
            all_valid_refs = chunk_ids | {t.tool_id for t in tools} | {m.memory_id for m in memory}
            invalid_citations = [r for r in cited_refs if r not in all_valid_refs]
            checks.append(CheckResult(
                name="citations_resolve",
                passed=len(invalid_citations) == 0,
                detail=f"Invalid: {invalid_citations}" if invalid_citations else "All citations valid",
            ))

            # no_dropped_cited: no dropped ref_id appears in prompt
            dropped_in_prompt = [r for r in dropped_ids if r in prompt]
            checks.append(CheckResult(
                name="no_dropped_cited",
                passed=len(dropped_in_prompt) == 0,
                detail=f"Dropped but cited: {dropped_in_prompt}" if dropped_in_prompt else "No dropped refs in prompt",
            ))

            # no_empty_keep: kept retrieval/memory blocks have non-empty text
            empty_keeps = [b.ref_id for b in ctx.blocks if b.kind in ("retrieval", "memory") and not b.text.strip()]
            checks.append(CheckResult(
                name="no_empty_keep",
                passed=len(empty_keeps) == 0,
                detail=f"Empty blocks kept: {empty_keeps}" if empty_keeps else "All kept blocks non-empty",
            ))

            # negative_retrieval: for negative intent, prompt should say "No passages retrieved" or have no retrieval citations
            if case.intent == "negative":
                no_retrieval = (
                    "No passages retrieved." in prompt or
                    not any(b.kind == "retrieval" for b in ctx.blocks)
                )
                checks.append(CheckResult(
                    name="negative_retrieval",
                    passed=no_retrieval,
                    detail="No retrieval for negative case" if no_retrieval else "Retriever returned hits for a negative case",
                ))

            elapsed_ms = (time.perf_counter() - t0) * 1000
            passed = all(c.passed for c in checks)
            trace.set_attributes(
                passed=passed,
                prompt_tokens=prompt_tokens,
                n_dropped=len(dropped_ids),
            )
            results.append(CaseResult(
                id=case.id,
                suite="assembly",
                passed=passed,
                checks=checks,
                cited_ids=cited_ids,
                prompt_tokens=prompt_tokens,
                latency_ms=elapsed_ms,
                trace_id=trace.trace_id,
            ))

    n = len(cases)
    sm = SuiteMetrics(
        n=n,
        n_pass=sum(1 for r in results if r.passed),
        n_fail=sum(1 for r in results if not r.passed),
        metrics={},
    )
    return results, sm

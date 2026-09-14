"""Answer suite — assemble + generate + grade.

Each case runs inside its own root trace named "eval.case" (assemble.pack,
retrieve.* and generate spans hang underneath), so a failed case can be
reopened with `python -m contextlab.trace show --case ans002`.
"""
import json
import os
import time
from pathlib import Path

from contextlab.evals.types import CaseResult, CheckResult, EvalCase, SuiteMetrics
from contextlab.assemble import assemble, BudgetError
from contextlab.evals.graders import (
    citation_support, required_facts, forbidden_facts,
    refusal_on_missing, judge_grader, grade_answer,
)
from contextlab.trace import prompt_attributes, start_span, start_trace


def load_golden(path: str | Path = "evals/answer_golden.jsonl") -> list[EvalCase]:
    """Load answer golden cases."""
    cases = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            d["suite"] = "answer"  # override suite from file
            cases.append(EvalCase(**d))
    return cases


def generate_answer(query: str, prompt: str, model: str = "gpt-4o-mini") -> tuple[str, float, float]:
    """Generate answer via OpenAI. Returns (output, latency_ms, cost_usd).

    Traced as the "generate" hop (model, token counts, latency_ms). The prompt
    is hashed, not stored: config/trace.yaml keeps include_prompt: false.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "", 0.0, 0.0

    try:
        from openai import OpenAI
    except ImportError:
        return "", 0.0, 0.0

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer questions using the provided context. Cite your sources using the format [chunk_id] after relevant information."},
        {"role": "user", "content": prompt},
    ]

    with start_span("generate", model=model) as gen_span:
        client = OpenAI(api_key=api_key)
        t0 = time.perf_counter()

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        output = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        gen_span.set_attributes(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            temperature=0.7,
            **prompt_attributes(prompt),
        )

    # Rough cost estimate: gpt-4o-mini ~ $0.15/1M input, $0.60/1M output
    cost_usd = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000
    return output, latency_ms, cost_usd


def run_suite(
    golden_path: str | Path = "evals/answer_golden.jsonl",
    offline: bool = False,
) -> tuple[list[CaseResult], SuiteMetrics]:
    """Run answer suite. Skips generation if offline or no API key."""
    cases = load_golden(golden_path)
    results: list[CaseResult] = []
    api_key = os.environ.get("OPENAI_API_KEY")
    has_key = bool(api_key)

    for case in cases:
        with start_trace(
            "eval.case", case_id=case.id, suite="answer", query=case.query, offline=offline
        ) as trace:
            t0 = time.perf_counter()
            checks: list[CheckResult] = []

            # Assemble context
            try:
                from contextlab.types import AssembleRequest, MemoryItem, ToolResult
                memory = [MemoryItem(**m) for m in case.memory_items]
                tools = [ToolResult(**t) for t in case.tool_results]
                request = AssembleRequest(
                    query=case.query,
                    system="You are a helpful assistant.",
                    budget_tokens=case.budget_tokens or 800,
                    retrieve=True,  # always retrieve so citation checks can run
                    k=5,
                    memory=memory,
                    tools=tools,
                )
                ctx = assemble(request)
                cited_ids = [b.ref_id for b in ctx.blocks if b.kind in ("retrieval", "memory")]
                prompt_tokens = ctx.prompt_tokens
            except BudgetError:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                trace.set_attributes(passed=False, budget_error=True)
                results.append(CaseResult(
                    id=case.id,
                    suite="answer",
                    passed=False,
                    checks=[CheckResult(name="budget_error", passed=False, detail="BudgetError during assembly")],
                    latency_ms=elapsed_ms,
                    trace_id=trace.trace_id,
                ))
                continue

            result = CaseResult(
                id=case.id,
                suite="answer",
                passed=False,
                cited_ids=cited_ids,
                prompt_tokens=prompt_tokens,
                latency_ms=0.0,
                trace_id=trace.trace_id,
            )

            # Generate answer (unless offline)
            if offline or not has_key:
                # Still run deterministic graders on the assembled prompt
                checks = grade_answer(case, result, generate_output=None)
                result.checks = checks
                result.passed = all(c.passed for c in checks)
                result.latency_ms = (time.perf_counter() - t0) * 1000
                trace.set_attributes(passed=result.passed, generated=False)
                results.append(result)
                continue

            # Generate with LLM
            generate_output, latency_ms, cost_usd = generate_answer(case.query, ctx.prompt)
            result.output = generate_output
            result.latency_ms = latency_ms
            result.cost_usd = cost_usd

            # Grade
            checks = grade_answer(case, result, generate_output=generate_output)
            if case.needs_judge and has_key:
                judge_result = judge_grader(case, result)
                checks.append(judge_result)
                result.judge_output = judge_result.model_dump() if hasattr(judge_result, "model_dump") else dict(judge_result)

            result.checks = checks
            result.passed = all(c.passed for c in checks)
            trace.set_attributes(passed=result.passed, generated=True, cost_usd=cost_usd)
            results.append(result)

    n = len(cases)
    n_skip = sum(1 for r in results if all(c.detail == "skipped_no_key" or c.detail == "needs_judge=false, skipped" for c in r.checks if c.name == "judge"))
    sm = SuiteMetrics(
        n=n,
        n_pass=sum(1 for r in results if r.passed),
        n_fail=sum(1 for r in results if not r.passed),
        n_skip=n_skip,
        metrics={},
    )
    return results, sm

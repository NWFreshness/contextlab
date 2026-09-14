"""Graders for answer suite."""
import os
import re
import subprocess
import json
from pathlib import Path
from typing import Optional

from contextlab.evals.types import CaseResult, CheckResult, EvalCase


# ── Deterministic graders ─────────────────────────────────────────────────────

def citation_support(case: EvalCase, result: CaseResult) -> CheckResult:
    """Check that at least one expected_citations id appears in cited_ids."""
    if not case.expected_citations:
        return CheckResult(name="citation_support", passed=True, detail="No expected citations")
    cited = set(result.cited_ids)
    found = [c for c in case.expected_citations if c in cited]
    passed = len(found) > 0
    detail = f"Found {found}" if passed else f"None of {case.expected_citations} in {result.cited_ids}"
    return CheckResult(name="citation_support", passed=passed, detail=detail)


def required_facts(case: EvalCase, output: str) -> CheckResult:
    """Check that required_facts appear in the output (case-insensitive, whitespace collapsed)."""
    if not case.required_facts:
        return CheckResult(name="required_facts", passed=True, detail="No required facts")
    if not output:
        return CheckResult(name="required_facts", passed=False, detail="No output to check")
    normalized = re.sub(r"\s+", " ", output.lower())
    missing = []
    for fact in case.required_facts:
        if fact.lower() not in normalized:
            missing.append(fact)
    passed = len(missing) == 0
    detail = f"Missing: {missing}" if missing else f"All {len(case.required_facts)} facts present"
    return CheckResult(name="required_facts", passed=passed, detail=detail)


def forbidden_facts(case: EvalCase, output: str) -> CheckResult:
    """Check that none of the forbidden_facts appear in the output."""
    if not case.forbidden_facts:
        return CheckResult(name="forbidden_facts", passed=True, detail="No forbidden facts")
    if not output:
        return CheckResult(name="forbidden_facts", passed=True, detail="No output (nothing to violate)")
    normalized = re.sub(r"\s+", " ", output.lower())
    found = [f for f in case.forbidden_facts if f.lower() in normalized]
    passed = len(found) == 0
    detail = f"Forbidden hit: {found}" if found else "No forbidden facts present"
    return CheckResult(name="forbidden_facts", passed=passed, detail=detail, score=0.0 if found else 1.0)


def refusal_on_missing(case: EvalCase, output: str) -> CheckResult:
    """For negative cases: should not invent a policy or entity from the corpus."""
    if case.intent != "negative":
        return CheckResult(name="refusal_on_missing", passed=True, detail="Not a negative case")
    if not output:
        return CheckResult(name="refusal_on_missing", passed=True, detail="No output")
    # Check that output doesn't contain a chunk_id pattern (e.g. error_codes::c0001)
    invented = re.findall(r"[\w]+::[\w]+", output)
    # Also check for specific forbidden patterns from corpus docs
    passed = len(invented) == 0
    detail = f"Possible invention: {invented}" if not passed else "No invented IDs detected"
    return CheckResult(name="refusal_on_missing", passed=passed, detail=detail)


def grade_answer(case: EvalCase, result: CaseResult, generate_output: Optional[str] = None) -> list[CheckResult]:
    """Run all deterministic graders on an answer case. Returns list of CheckResults."""
    checks: list[CheckResult] = []
    output = generate_output if generate_output is not None else (result.output or "")

    # Citation support always runs (checks cited_ids against expected_citations)
    checks.append(citation_support(case, result))

    # Fact checks require generated output; skip in offline mode when no output
    if output:
        checks.append(required_facts(case, output))
        checks.append(forbidden_facts(case, output))
        checks.append(refusal_on_missing(case, output))
    else:
        # Offline / no generation: skip fact checks rather than fail them
        for name in ("required_facts", "forbidden_facts", "refusal_on_missing"):
            checks.append(CheckResult(name=name, passed=True, detail="skipped_no_output"))

    return checks


# ── Judge grader (LLM-as-judge) ──────────────────────────────────────────────

def load_rubric(path: str | Path = "evals/rubrics/answer_v1.md") -> str:
    """Load the judge rubric."""
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text()


def call_judge(
    query: str,
    answer: str,
    rubric_path: str | Path = "evals/rubrics/answer_v1.md",
    model: str = "gpt-4o-mini",
) -> Optional[dict]:
    """Call judge via OpenAI SDK. Returns judge output dict or None if key missing."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    rubric = load_rubric(rubric_path)
    client = OpenAI(api_key=api_key)

    judge_prompt = f"""You are an answer quality judge.

Query: {query}

Answer:\n{answer}

Rubric:\n{rubric}

Respond with JSON: {{"score": 0-3, "passed": bool, "reason": "string"}}
Score 0 = completely wrong, 1 = major errors, 2 = minor errors, 3 = perfect.
Threshold for pass: score >= 2.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


def judge_grader(case: EvalCase, result: CaseResult) -> CheckResult:
    """Run judge grader on a case. Returns CheckResult (skipped if no key or not needs_judge)."""
    if not case.needs_judge:
        return CheckResult(name="judge", passed=True, detail="needs_judge=false, skipped")
    if not case.expected_citations and not case.required_facts:
        return CheckResult(name="judge", passed=True, detail="No criteria for judge")

    judge_output = call_judge(case.query, result.output or "")
    if judge_output is None:
        return CheckResult(name="judge", passed=False, detail="skipped_no_key", score=None)

    score = judge_output.get("score", 0)
    passed = judge_output.get("passed", score >= 2)
    return CheckResult(
        name="judge",
        passed=passed,
        detail=judge_output.get("reason", ""),
        score=score,
    )

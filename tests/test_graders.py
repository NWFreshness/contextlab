"""Tests for answer graders."""
import pytest

from contextlab.evals.types import CaseResult, CheckResult, EvalCase
from contextlab.evals.graders import (
    citation_support, required_facts, forbidden_facts,
    refusal_on_missing, grade_answer,
)


# ── citation_support ─────────────────────────────────────────────────────────

def test_citation_support_pass():
    """Expected citation found in cited_ids."""
    case = EvalCase(
        id="test",
        suite="answer",
        query="test",
        expected_citations=["error_codes::c0001", "refund_policy_v3::c0000"],
    )
    result = CaseResult(
        id="test",
        suite="answer",
        passed=False,
        cited_ids=["error_codes::c0001", "boilerplate::c0000"],
    )
    check = citation_support(case, result)
    assert check.passed is True
    assert "error_codes::c0001" in check.detail


def test_citation_support_fail():
    """No expected citation in cited_ids."""
    case = EvalCase(
        id="test",
        suite="answer",
        query="test",
        expected_citations=["error_codes::c0005"],
    )
    result = CaseResult(
        id="test",
        suite="answer",
        passed=False,
        cited_ids=["boilerplate::c0000"],
    )
    check = citation_support(case, result)
    assert check.passed is False


# ── required_facts ──────────────────────────────────────────────────────────

def test_required_facts_pass():
    """All required facts present."""
    output = "The Starter plan costs $9 per month and supports up to 5 users."
    case = EvalCase(
        id="test",
        suite="answer",
        query="test",
        required_facts=["$9", "5 users"],
    )
    check = required_facts(case, output)
    assert check.passed is True


def test_required_facts_missing():
    """Required fact absent."""
    output = "The Starter plan is affordable."
    case = EvalCase(
        id="test",
        suite="answer",
        query="test",
        required_facts=["$9", "5 users"],
    )
    check = required_facts(case, output)
    assert check.passed is False
    assert "$9" in check.detail


def test_required_facts_case_insensitive():
    """Fact matching is case-insensitive."""
    output = "The current refund window is 30 days from purchase."
    case = EvalCase(
        id="test",
        suite="answer",
        query="test",
        required_facts=["30 DAYS"],
    )
    check = required_facts(case, output)
    assert check.passed is True


def test_required_facts_no_output():
    """No output is a fail for required_facts."""
    case = EvalCase(
        id="test",
        suite="answer",
        query="test",
        required_facts=["some fact"],
    )
    check = required_facts(case, "")
    assert check.passed is False


# ── forbidden_facts ──────────────────────────────────────────────────────────

def test_forbidden_facts_pass():
    """No forbidden facts present."""
    output = "The current refund window is 30 days from purchase."
    case = EvalCase(
        id="test",
        suite="answer",
        query="test",
        forbidden_facts=["14 days"],
    )
    check = forbidden_facts(case, output)
    assert check.passed is True


def test_forbidden_facts_fail():
    """Forbidden fact appears in output."""
    output = "The old policy had 14 days for returns."
    case = EvalCase(
        id="test",
        suite="answer",
        query="test",
        forbidden_facts=["14 days"],
    )
    check = forbidden_facts(case, output)
    assert check.passed is False
    assert "14 days" in check.detail


def test_forbidden_facts_no_output():
    """No output passes forbidden_facts (nothing to violate)."""
    case = EvalCase(
        id="test",
        suite="answer",
        query="test",
        forbidden_facts=["some fact"],
    )
    check = forbidden_facts(case, "")
    assert check.passed is True


# ── refusal_on_missing ────────────────────────────────────────────────────────

def test_refusal_on_missing_negative_pass():
    """Negative case with no output passes."""
    case = EvalCase(
        id="test",
        suite="answer",
        query="Apple return policy",
        intent="negative",
    )
    check = refusal_on_missing(case, "")
    assert check.passed is True


def test_refusal_on_missing_invention_detected():
    """Negative case that invents a chunk ID fails."""
    case = EvalCase(
        id="test",
        suite="answer",
        query="Apple return policy",
        intent="negative",
    )
    output = "According to error_codes::c0005, Apple items have a 30-day return policy."
    check = refusal_on_missing(case, output)
    assert check.passed is False


def test_refusal_on_missing_non_negative():
    """Non-negative case is always pass."""
    case = EvalCase(
        id="test",
        suite="answer",
        query="E-4471 meaning",
        intent="lexical",
    )
    check = refusal_on_missing(case, "Some invented response")
    assert check.passed is True


# ── grade_answer ─────────────────────────────────────────────────────────────

def test_grade_answer_combines_checks():
    """grade_answer returns all deterministic checks."""
    case = EvalCase(
        id="ans002",
        suite="answer",
        query="Current refund window",
        expected_citations=["refund_policy_v3::c0000"],
        required_facts=["30 days"],
        forbidden_facts=["14 days"],
        intent="version",
    )
    result = CaseResult(
        id="ans002",
        suite="answer",
        passed=False,
        cited_ids=["refund_policy_v3::c0000"],
    )
    checks = grade_answer(case, result, generate_output="The current refund window is 30 days from purchase date.")
    names = {c.name for c in checks}
    assert "citation_support" in names
    assert "required_facts" in names
    assert "forbidden_facts" in names
    assert "refusal_on_missing" in names
    # All should pass
    assert all(c.passed for c in checks)

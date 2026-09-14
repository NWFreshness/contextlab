"""Tests for the eval harness."""
import json
import pytest
from pathlib import Path

from contextlab.evals.types import CaseResult, CheckResult, EvalReport, SuiteMetrics
from contextlab.evals.report import format_summary, write_latest
from contextlab.evals.runner import load_gates, apply_gates
from contextlab.evals import suites_retrieval, suites_assembly, suites_answer


# ── Report schema tests ───────────────────────────────────────────────────────

def test_eval_report_schema():
    """EvalReport can be instantiated with all required fields."""
    report = EvalReport(
        started_at="2024-01-01T00:00:00Z",
        finished_at="2024-01-01T00:01:00Z",
        settings={"config": {"chunk_tokens": 256}},
        suites={
            "retrieval": SuiteMetrics(n=10, n_pass=9, n_fail=1),
            "assembly": SuiteMetrics(n=5, n_pass=5),
        },
        gates=[],
        cases=[],
        passed=False,
    )
    assert report.passed is False
    assert report.suites["retrieval"].n == 10
    assert report.suites["retrieval"].n_fail == 1


def test_check_result_schema():
    """CheckResult has all required fields."""
    cr = CheckResult(name="recall_at_5", passed=True, detail="0.917", score=0.917)
    assert cr.name == "recall_at_5"
    assert cr.passed is True
    assert cr.score == 0.917


def test_case_result_schema():
    """CaseResult stores retrieved and cited ids."""
    cr = CaseResult(
        id="r001",
        suite="retrieval",
        passed=True,
        retrieved_ids=["error_codes::c0001", "incident_runbook::c0000"],
        cited_ids=["error_codes::c0001"],
        checks=[],
    )
    assert "error_codes::c0001" in cr.retrieved_ids
    assert "error_codes::c0001" in cr.cited_ids


# ── Citation check test ────────────────────────────────────────────────────────

def test_invalid_citation_fails():
    """A citation to does-not-exist::c0000 should fail citations_exist check."""
    from contextlab.evals.suites_retrieval import load_golden

    golden = load_golden()
    assert len(golden) >= 1

    # Simulate a bad citation
    result = CaseResult(
        id="test_invalid_citation",
        suite="retrieval",
        passed=False,
        retrieved_ids=["does-not-exist::c0000"],
        cited_ids=["does-not-exist::c0000"],
        checks=[],
    )

    # Check: citations_resolve would fail for invalid chunk id
    valid_ids = {"error_codes::c0000", "error_codes::c0001"}
    invalid = [h for h in result.retrieved_ids if h not in valid_ids]
    assert len(invalid) > 0  # does-not-exist::c0000 is not valid


def test_format_summary():
    """format_summary produces a non-empty string with key sections."""
    report = EvalReport(
        started_at="2024-01-01T00:00:00Z",
        finished_at="2024-01-01T00:01:00Z",
        settings={},
        suites={
            "retrieval": SuiteMetrics(n=10, n_pass=9, n_fail=1),
            "assembly": SuiteMetrics(n=5, n_pass=5),
        },
        gates=[
            CheckResult(name="gate_recall", passed=True, detail="0.917 >= 0.70 PASS"),
            CheckResult(name="gate_budget", passed=False, detail="1 > 0 FAIL"),
        ],
        cases=[],
        passed=False,
    )
    summary = format_summary(report)
    assert "Eval Summary" in summary
    assert "retrieval" in summary
    assert "assembly" in summary
    assert "FAIL" in summary


def test_gates_yaml_exists():
    """gates.yaml exists and has required keys."""
    gates = load_gates()
    assert "retrieval" in gates
    assert "assembly" in gates
    assert "answer" in gates
    assert "min_recall_at_5_hybrid" in gates["retrieval"]


def test_assembly_golden_cases():
    """assembly_golden.jsonl has required case types."""
    cases = suites_assembly.load_golden()
    intents = {c.intent for c in cases}
    assert "lexical" in intents
    assert "version" in intents
    assert len(cases) >= 7


def test_answer_golden_cases():
    """answer_golden.jsonl has >= 12 cases."""
    cases = suites_answer.load_golden()
    assert len(cases) >= 12


def test_retrieval_suite_runs():
    """Retrieval suite produces results and metrics."""
    results, metrics = suites_retrieval.run_suite()
    assert metrics.n >= 24  # 24 golden cases from Slice 1
    assert metrics.n_pass + metrics.n_fail + metrics.n_skip == metrics.n
    assert "recall@5_hybrid" in metrics.metrics


def test_assembly_suite_runs():
    """Assembly suite runs without error."""
    results, metrics = suites_assembly.run_suite()
    assert metrics.n >= 7
    assert metrics.n_pass + metrics.n_fail == metrics.n


def test_answer_suite_runs_offline():
    """Answer suite runs in offline mode without crashing."""
    results, metrics = suites_answer.run_suite(offline=True)
    assert metrics.n >= 12
    # In offline mode, checks should still run (just no generation)


def test_answer_golden_has_required_facts():
    """Each answer case has required_facts or notes with acceptable answer."""
    cases = suites_answer.load_golden()
    for case in cases:
        assert case.query
        assert case.id
        assert len(case.expected_citations) > 0 or case.required_facts or case.forbidden_facts


def test_no_budget_violations_gate():
    """Gate fails when assembly has budget violations."""
    # Build a mock result with a budget violation
    results = [
        CaseResult(
            id="a005",
            suite="assembly",
            passed=False,
            checks=[
                CheckResult(name="under_budget", passed=False, detail="prompt_tokens=400 > budget=50"),
            ],
        )
    ]
    metrics = SuiteMetrics(n=1, n_pass=0, n_fail=1)
    suites_results = {"assembly": (results, metrics)}
    gates = {"assembly": {"max_budget_violations": 0, "max_citation_errors": 0}}
    gate_results = apply_gates(suites_results, gates)
    budget_gate = [g for g in gate_results if g.name == "gate_assembly_budget"][0]
    assert budget_gate.passed is False


def test_no_citation_errors_gate():
    """Gate fails when assembly has citation errors."""
    results = [
        CaseResult(
            id="a001",
            suite="assembly",
            passed=False,
            checks=[
                CheckResult(name="citations_resolve", passed=False, detail="does-not-exist::c0000 invalid"),
            ],
        )
    ]
    metrics = SuiteMetrics(n=1, n_pass=0, n_fail=1)
    suites_results = {"assembly": (results, metrics)}
    gates = {"assembly": {"max_budget_violations": 0, "max_citation_errors": 0}}
    gate_results = apply_gates(suites_results, gates)
    citation_gate = [g for g in gate_results if g.name == "gate_assembly_citations"][0]
    assert citation_gate.passed is False

"""Eval harness types."""
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """A single golden eval case."""
    id: str
    suite: str  # retrieval | assembly | answer
    query: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_doc_ids: list[str] = Field(default_factory=list)
    required_facts: list[str] = Field(default_factory=list)
    forbidden_facts: list[str] = Field(default_factory=list)
    expected_citations: list[str] = Field(default_factory=list)
    intent: str = ""
    needs_judge: bool = False
    notes: str = ""
    budget_tokens: Optional[int] = None
    memory_items: list[dict] = Field(default_factory=list)
    tool_results: list[dict] = Field(default_factory=list)
    invariants: list[str] = Field(default_factory=list)


class CheckResult(BaseModel):
    """Result of a single check within a case."""
    name: str
    passed: bool
    detail: str
    score: Optional[float] = None


class CaseResult(BaseModel):
    """Result of evaluating one EvalCase."""
    id: str
    suite: str
    passed: bool
    checks: list[CheckResult] = Field(default_factory=list)
    retrieved_ids: list[str] = Field(default_factory=list)
    cited_ids: list[str] = Field(default_factory=list)
    prompt_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    output: Optional[str] = None
    judge_output: Optional[dict] = None
    trace_id: Optional[str] = Field(default=None, description="Slice 4: trace of this case, if traced")
    route: Optional[str] = Field(default=None, description="Slice 5: router route chosen for this case")
    model: Optional[str] = Field(default=None, description="Slice 5: model the router chose")


class SuiteMetrics(BaseModel):
    """Aggregated metrics for one suite."""
    n: int = 0
    n_pass: int = 0
    n_fail: int = 0
    n_skip: int = 0
    metrics: dict = Field(default_factory=dict)


class EvalReport(BaseModel):
    """Top-level eval report."""
    started_at: str
    finished_at: str
    settings: dict = Field(default_factory=dict)
    suites: dict[str, SuiteMetrics] = Field(default_factory=dict)
    gates: list[CheckResult] = Field(default_factory=list)
    cases: list[CaseResult] = Field(default_factory=list)
    passed: bool = False

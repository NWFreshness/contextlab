"""Eval harness types."""
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """A single golden eval case."""
    id: str
    suite: str  # retrieval | assembly | answer | trajectory | ...
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


class TrajectoryCase(EvalCase):
    """Slice 9 — golden row for trajectory grading.

    Additive fields; defaults keep Slices 3–6 suites parsing unchanged
    when an old row reaches this loader (they don't, but a typo here
    shouldn't break an unrelated suite).

    `policy_name` selects a fixture policy from `script_policy.fixtures`
    when present (`always_retrieve`, `unknown_tool`, `arithmetic_only`,
    `identifier_only`). `tools` overrides the orchestrator's registered
    tools. `settings_override` merges into `OrchRequest.settings` so a
    case can shrink `max_steps`, change `retrieve_k`, etc.

    Constraints are interpreted by `graders_trajectory`. Each constraint
    is independent — a case fails if any of its active checks fail.
    """

    suite: str = "trajectory"
    # Per-case knobs (override the orchestrator defaults)
    policy_name: str = ""                   # empty = the production ScriptedPolicy
    driver: str = "script"                  # script (default) | llm | fixture
    tools: list[str] = Field(default_factory=list)  # override OrchRequest.tools
    settings_override: dict = Field(default_factory=dict)
    # Constraint fields
    expected_stop_reasons: list[str] = Field(default_factory=lambda: ["done"])
    max_steps_cap: Optional[int] = None      # explicit cap; <= orchestrator max_steps
    required_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    tool_order_strict: bool = False
    required_observation_substrings: list[str] = Field(default_factory=list)
    forbidden_observation_substrings: list[str] = Field(default_factory=list)
    required_final_answer_substrings: list[str] = Field(default_factory=list)
    forbidden_final_answer_substrings: list[str] = Field(default_factory=list)
    needs_llm: bool = False                 # skip offline runs
    # Case type — used by the gate to identify happy-path cases
    case_type: str = "happy"                 # happy | max_steps | unknown_tool | policy | timeout


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

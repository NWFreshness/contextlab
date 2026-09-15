"""Trace record types — OpenTelemetry-shaped, JSON-safe.

Field names mirror the OTel span model (trace_id / span_id / parent_span_id /
name / start_ns / end_ns / status / error / attributes) so a later OTel
exporter is a thin adapter over `SpanRecord` rather than a rewrite. Nothing
here talks to a collector: the default store is JSONL under data/traces/.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field

# id shapes: 32 hex chars for a trace, 16 hex chars for a span
TRACE_ID_HEX = 32
SPAN_ID_HEX = 16

# Required attributes by hop. "*" matches a hop family: "retrieve.*" covers
# retrieve.bm25 / retrieve.dense / retrieve.fuse / retrieve.hybrid.
# A span that closes without these gets attributes["missing_context"] = [...]
# rather than silently looking complete.
REQUIRED_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "ingest": ("n_docs", "n_chunks", "doc_ids", "chunk_ids"),
    "retrieve.*": ("query", "mode", "k", "chunk_ids", "doc_ids"),
    "assemble.pack": ("budget_tokens", "prompt_tokens", "kept_ids", "dropped_ids"),
    "generate": ("model", "input_tokens", "output_tokens", "latency_ms"),
    "eval.case": ("case_id", "suite", "passed"),
    "router.decide": ("route", "model", "reason", "intent_hint", "has_identifier", "dry_run"),
    "cache.lookup": ("hit", "score", "reason"),
    "cache.write": ("cache_id",),
    "tool.exec": ("tool", "backend", "timeout_s"),
}


class SpanRecord(BaseModel):
    """One span, exactly as it is written to JSONL."""

    trace_id: str = Field(description=f"{TRACE_ID_HEX} lowercase hex chars")
    span_id: str = Field(description=f"{SPAN_ID_HEX} lowercase hex chars")
    parent_span_id: Optional[str] = Field(default=None, description="None on the root span")
    name: str = Field(description='hop name, e.g. "retrieve.hybrid"')
    start_ns: int = Field(description="wall clock, nanoseconds")
    end_ns: Optional[int] = Field(default=None, description="None while the span is open")
    status: Literal["ok", "error"] = "ok"
    error: Optional[str] = None
    attributes: dict = Field(default_factory=dict, description="JSON-safe primitives + list[str]")

    @property
    def duration_ms(self) -> Optional[float]:
        """Span duration in milliseconds, or None while the span is open."""
        if self.end_ns is None:
            return None
        return (self.end_ns - self.start_ns) / 1_000_000

    def to_json_line(self) -> str:
        """One JSON object per line — the JSONL wire format."""
        return self.model_dump_json()


class TraceSummary(BaseModel):
    """Roll-up of one trace. Derived from SpanRecords; stored nowhere."""

    trace_id: str
    root_name: str
    duration_ms: float
    span_count: int
    chunk_ids: list[str] = Field(default_factory=list)
    dropped_ids: list[str] = Field(default_factory=list)
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    # additive convenience for `trace show`: how many spans ended in error
    n_error: int = 0


def required_attributes(span_name: str) -> tuple[str, ...]:
    """Required attribute names for a span name (() when the hop is unknown)."""
    for hop, names in REQUIRED_ATTRIBUTES.items():
        if hop.endswith(".*"):
            if span_name.startswith(hop[:-1]):
                return names
        elif span_name == hop:
            return names
    return ()


def missing_required_attributes(record: SpanRecord) -> list[str]:
    """Required attributes absent from a span — the contract check for a hop."""
    return [name for name in required_attributes(record.name) if name not in record.attributes]

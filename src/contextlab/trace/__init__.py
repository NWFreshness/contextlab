"""Slice 4 — LLM tracer: OTel-shaped spans, JSONL store, reconstructable hops.

    from contextlab.trace import start_trace

    with start_trace("assemble") as trace:
        with start_span("retrieve.hybrid", query="...", mode="hybrid", k=8) as span:
            span.set_attributes(chunk_ids=[...], doc_ids=[...])
        trace_id = trace.trace_id

Public API: start_trace / start_span / configure / reset / current_trace_id,
the SpanRecord + TraceSummary contracts, and the exporters.
"""
from contextlab.trace import show  # noqa: F401  (kept importable as contextlab.trace.show)
from contextlab.trace.export import (
    InMemoryExporter,
    JSONLExporter,
    NullExporter,
    SpanExporter,
    build_summary,
    chunk_ids_for,
    dropped_ids_for,
    find_latest_trace_id,
    find_trace_id_for_case,
    group_by_trace,
    load_records,
    trace_dir,
)
from contextlab.trace.tracer import (
    Span,
    Trace,
    Tracer,
    TracerConfig,
    configure,
    current_span,
    current_trace_id,
    get_config,
    get_exporter,
    get_tracer,
    key_is_secret,
    prompt_attributes,
    redact_record,
    reset,
    start_span,
    start_trace,
)
from contextlab.trace.types import (
    REQUIRED_ATTRIBUTES,
    SPAN_ID_HEX,
    TRACE_ID_HEX,
    SpanRecord,
    TraceSummary,
    missing_required_attributes,
    required_attributes,
)

__all__ = [
    "InMemoryExporter",
    "JSONLExporter",
    "NullExporter",
    "REQUIRED_ATTRIBUTES",
    "SPAN_ID_HEX",
    "TRACE_ID_HEX",
    "Span",
    "SpanExporter",
    "SpanRecord",
    "Trace",
    "TraceSummary",
    "Tracer",
    "TracerConfig",
    "build_summary",
    "chunk_ids_for",
    "configure",
    "current_span",
    "current_trace_id",
    "dropped_ids_for",
    "find_latest_trace_id",
    "find_trace_id_for_case",
    "get_config",
    "get_exporter",
    "get_tracer",
    "group_by_trace",
    "key_is_secret",
    "load_records",
    "missing_required_attributes",
    "prompt_attributes",
    "redact_record",
    "required_attributes",
    "reset",
    "show",
    "start_span",
    "start_trace",
    "trace_dir",
]

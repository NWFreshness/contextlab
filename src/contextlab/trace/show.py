"""`python -m contextlab.trace show` — resolve a trace and render its span tree."""
import json
from datetime import datetime, timezone
from typing import Optional, Sequence

from contextlab.trace.export import (
    build_summary,
    group_by_trace,
    find_latest_trace_id,
    find_trace_id_for_case,
)
from contextlab.trace.types import SpanRecord, TraceSummary


def select_trace_id(
    records: Sequence[SpanRecord],
    trace_id: Optional[str] = None,
    case_id: Optional[str] = None,
    last: bool = False,
) -> Optional[str]:
    """Resolve which trace to show: explicit id > case id > most recent."""
    known = set(group_by_trace(records))
    if trace_id:
        return trace_id if trace_id in known else None
    if case_id:
        return find_trace_id_for_case(records, case_id)
    if last:
        return find_latest_trace_id(records)
    return None


def spans_of(records: Sequence[SpanRecord], trace_id: str) -> list[SpanRecord]:
    """Spans of one trace, oldest start first."""
    return sorted((r for r in records if r.trace_id == trace_id), key=lambda r: r.start_ns)


def iso_from_ns(ns: int) -> str:
    """Wall-clock ISO-8601 (UTC, milliseconds) for a nanosecond timestamp."""
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat(timespec="milliseconds")


def _format_value(value, max_value: int, full: bool) -> str:
    if isinstance(value, list):
        items = [str(v) for v in value]
        if full or len(items) <= 3:
            rendered = ", ".join(items)
        else:
            rendered = ", ".join(items[:3]) + f", …(+{len(items) - 3})"
        text = f"[{rendered}]"
    else:
        text = json.dumps(value)
    if not full and len(text) > max_value:
        text = text[: max_value - 1] + "…"
    return text


def format_attributes(attributes: dict, max_value: int = 60, full: bool = False) -> str:
    """`key=value` pairs, one line, truncated unless --full."""
    parts = [f"{key}={_format_value(value, max_value, full)}" for key, value in attributes.items()]
    return "  ".join(parts)


def render_tree(spans: Sequence[SpanRecord], full: bool = False) -> str:
    """Indented span tree — parentage read back from parent_span_id."""
    by_id = {s.span_id: s for s in spans}
    children: dict[Optional[str], list[SpanRecord]] = {}
    for span in spans:
        parent_id = span.parent_span_id if span.parent_span_id in by_id else None
        children.setdefault(parent_id, []).append(span)

    lines: list[str] = []
    width = min(max((len(s.name) for s in spans), default=10) + 2, 34)

    def walk(span: SpanRecord, depth: int) -> None:
        duration = f"{span.duration_ms:8.1f} ms" if span.duration_ms is not None else "   open   "
        flag = ""
        if span.status == "error":
            flag = f"  ERROR: {span.error}"
        elif "missing_context" in span.attributes:
            flag = f"  MISSING: {span.attributes['missing_context']}"

        # missing_context is already surfaced as a MISSING flag on the span line
        attrs = {
            key: value
            for key, value in span.attributes.items()
            if key != "missing_context" or full
        }
        line = f"{'  ' * depth}{span.name:<{width}} {duration}  {span.status:<5}{flag}"
        lines.append(line)
        attr_line = format_attributes(attrs, full=full)
        if attr_line:
            lines.append(f"{'  ' * depth}{' ' * (width + 2)}  {attr_line}")

        for child in sorted(children.get(span.span_id, []), key=lambda s: s.start_ns):
            walk(child, depth + 1)

    for root in sorted(children.get(None, []), key=lambda s: s.start_ns):
        walk(root, 0)

    return "\n".join(lines)


def render_summary(summary: TraceSummary) -> str:
    """Header block: what this trace was, how long it took, what was in context."""
    lines = [
        f"=== Trace {summary.trace_id} ===",
        f"root         {summary.root_name}",
        f"duration     {summary.duration_ms:.1f} ms",
        f"spans        {summary.span_count}  ({summary.n_error} in error)",
        f"chunk_ids    [{len(summary.chunk_ids)}] {', '.join(summary.chunk_ids) if summary.chunk_ids else '-'}",
        f"dropped_ids  [{len(summary.dropped_ids)}] {', '.join(summary.dropped_ids) if summary.dropped_ids else '-'}",
        f"model        {summary.model or '-'}",
        f"prompt_tokens {summary.prompt_tokens if summary.prompt_tokens is not None else '-'}",
    ]
    return "\n".join(lines)


def render_trace(spans: Sequence[SpanRecord], full: bool = False) -> str:
    """Summary + span tree for one trace."""
    summary = build_summary(list(spans))
    return "\n".join(
        [
            render_summary(summary),
            f"started      {iso_from_ns(min(s.start_ns for s in spans))}",
            "",
            "span tree",
            render_tree(spans, full=full),
        ]
    )


def render_trace_list(records: Sequence[SpanRecord], limit: int = 20) -> str:
    """Recent traces, newest first — trace_id, root, size, duration."""
    rows: list[tuple[int, str, str, int, float, int]] = []
    for trace_id, spans in group_by_trace(records).items():
        summary = build_summary(spans)
        latest = max(s.start_ns for s in spans)
        rows.append(
            (latest, trace_id, summary.root_name, summary.span_count, summary.duration_ms, summary.n_error)
        )
    rows.sort(reverse=True)

    if not rows:
        return "no traces found"

    lines = [f"{'trace_id':<34}{'root':<18}{'spans':>6}{'ms':>10}{'err':>5}  started"]
    for latest, trace_id, root_name, count, duration_ms, n_error in rows[:limit]:
        lines.append(
            f"{trace_id:<34}{root_name:<18}{count:>6}{duration_ms:>10.1f}{n_error:>5}  {iso_from_ns(latest)}"
        )
    return "\n".join(lines)

"""Span exporters and JSONL query helpers.

The default exporter appends one JSON object per span to a per-day JSONL file
(data/traces/YYYYMMDD.jsonl). Tests swap in InMemoryExporter so they never
touch the repo's trace files.
"""
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from contextlab.trace.types import SpanRecord, TraceSummary


class SpanExporter:
    """Base class: anything that can receive finished spans."""

    def export(self, records: list[SpanRecord]) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class NullExporter(SpanExporter):
    """Drops everything (tracing disabled)."""

    def export(self, records: list[SpanRecord]) -> None:
        return None


class InMemoryExporter(SpanExporter):
    """Keeps spans in a list. Used by tests."""

    def __init__(self) -> None:
        self.records: list[SpanRecord] = []

    def export(self, records: list[SpanRecord]) -> None:
        self.records.extend(records)

    def clear(self) -> None:
        self.records.clear()

    def for_trace(self, trace_id: str) -> list[SpanRecord]:
        """Spans of one trace, in export order."""
        return [r for r in self.records if r.trace_id == trace_id]

    def trace_ids(self) -> list[str]:
        """Trace ids in export order, de-duplicated."""
        seen: OrderedDict[str, None] = OrderedDict()
        for r in self.records:
            seen.setdefault(r.trace_id, None)
        return list(seen)


class JSONLExporter(SpanExporter):
    """Append spans to <directory>/YYYYMMDD.jsonl — grep-able, no migrations."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def path_for(self, when: Optional[datetime] = None) -> Path:
        """File the next export lands in."""
        when = when or datetime.now(timezone.utc)
        return self.directory / f"{when.strftime('%Y%m%d')}.jsonl"

    def export(self, records: list[SpanRecord]) -> None:
        if not records:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for()
        with open(path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(record.to_json_line() + "\n")


# ── Reading traces back ───────────────────────────────────────────────────────


def trace_dir(directory: str | Path | None = None) -> Path:
    """Resolve the trace directory (explicit arg > tracer config)."""
    if directory is not None:
        return Path(directory)
    from contextlab.trace.tracer import get_config  # deferred: avoids import cycle

    return get_config().directory


def load_records(directory: str | Path | None = None) -> list[SpanRecord]:
    """Read every JSONL trace file oldest-name-first. Unparseable lines are skipped."""
    root = trace_dir(directory)
    if not root.exists():
        return []

    records: list[SpanRecord] = []
    for path in sorted(root.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(SpanRecord(**json.loads(line)))
                except (json.JSONDecodeError, ValueError):
                    continue  # partial line from an interrupted write
    return records


def group_by_trace(records: Iterable[SpanRecord]) -> OrderedDict[str, list[SpanRecord]]:
    """{trace_id: [spans in file order]}."""
    grouped: OrderedDict[str, list[SpanRecord]] = OrderedDict()
    for record in records:
        grouped.setdefault(record.trace_id, []).append(record)
    return grouped


def _root_of(spans: list[SpanRecord]) -> SpanRecord:
    """The root span of a trace (no parent), falling back to the earliest span."""
    for span in spans:
        if span.parent_span_id is None:
            return span
    return min(spans, key=lambda s: s.start_ns)


def find_latest_trace_id(records: Iterable[SpanRecord]) -> Optional[str]:
    """Trace id whose root span started most recently."""
    best_id: Optional[str] = None
    best_start = -1
    for trace_id, spans in group_by_trace(records).items():
        start = _root_of(spans).start_ns
        if start >= best_start:
            best_start = start
            best_id = trace_id
    return best_id


def find_trace_id_for_case(records: Iterable[SpanRecord], case_id: str) -> Optional[str]:
    """Latest trace containing an eval.case span for this case id."""
    best_id: Optional[str] = None
    best_start = -1
    for record in records:
        if record.name != "eval.case":
            continue
        if str(record.attributes.get("case_id")) != str(case_id):
            continue
        if record.start_ns >= best_start:
            best_start = record.start_ns
            best_id = record.trace_id
    return best_id


def _is_outermost_retrieve(record: SpanRecord, by_id: dict[str, SpanRecord]) -> bool:
    """True when no ancestor of this span is itself a retrieve hop.

    Keeps the summary honest: the bm25 sub-span's 50 candidates must not be
    reported as "the chunk ids retrieve returned".
    """
    parent_id = record.parent_span_id
    seen: set[str] = set()
    while parent_id and parent_id not in seen:
        seen.add(parent_id)
        parent = by_id.get(parent_id)
        if parent is None:
            return True
        if parent.name.startswith("retrieve."):
            return False
        parent_id = parent.parent_span_id
    return True


def chunk_ids_for(spans: list[SpanRecord]) -> list[str]:
    """Chunk ids returned by the outermost retrieve hop(s), in order, de-duplicated."""
    by_id = {s.span_id: s for s in spans}
    ordered: OrderedDict[str, None] = OrderedDict()
    for span in sorted(spans, key=lambda s: s.start_ns):
        if not span.name.startswith("retrieve.") or not _is_outermost_retrieve(span, by_id):
            continue
        for chunk_id in span.attributes.get("chunk_ids") or []:
            ordered.setdefault(str(chunk_id), None)
    return list(ordered)


def dropped_ids_for(spans: list[SpanRecord]) -> list[str]:
    """Refs the packer dropped (from assemble.pack spans), in order, de-duplicated."""
    ordered: OrderedDict[str, None] = OrderedDict()
    for span in sorted(spans, key=lambda s: s.start_ns):
        if span.name != "assemble.pack":
            continue
        for ref_id in span.attributes.get("dropped_ids") or []:
            ordered.setdefault(str(ref_id), None)
    return list(ordered)


def build_summary(spans: list[SpanRecord]) -> TraceSummary:
    """Roll a trace's spans up into a TraceSummary."""
    root = _root_of(spans)
    if root.end_ns is not None:
        duration_ms = (root.end_ns - root.start_ns) / 1_000_000
    else:
        end = max((s.end_ns or s.start_ns) for s in spans)
        duration_ms = (end - min(s.start_ns for s in spans)) / 1_000_000

    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    for span in sorted(spans, key=lambda s: s.start_ns):
        if span.name == "generate" and span.attributes.get("model") is not None:
            model = str(span.attributes["model"])
        if span.name == "assemble.pack" and span.attributes.get("prompt_tokens") is not None:
            prompt_tokens = int(span.attributes["prompt_tokens"])

    return TraceSummary(
        trace_id=root.trace_id,
        root_name=root.name,
        duration_ms=duration_ms,
        span_count=len(spans),
        chunk_ids=chunk_ids_for(spans),
        dropped_ids=dropped_ids_for(spans),
        model=model,
        prompt_tokens=prompt_tokens,
        n_error=sum(1 for s in spans if s.status == "error"),
    )

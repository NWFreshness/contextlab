"""Slice 4 — hop tracer: contextvar current span + JSONL exporter.

Records are shaped like OpenTelemetry spans (trace_id / span_id /
parent_span_id / name / start_ns / end_ns / status / error / attributes) but no
collector is required. The default store is one JSONL file per day under
data/traces/.

Span hops (deliberately not every function):
    ingest, retrieve.bm25, retrieve.dense, retrieve.fuse, assemble.pack,
    generate, eval.case

Tracing is opt-in per trace: spans are only recorded while a trace started by
start_trace() is active on the current context. The CLI entry points
(retrieve / assemble / ingest / evals) start that root trace; a library caller
that never asked for a trace pays one contextvar lookup per span site.
"""
from __future__ import annotations

import contextvars
import hashlib
import os
import re
import secrets
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import yaml

from contextlab.trace.export import JSONLExporter, NullExporter, SpanExporter
from contextlab.trace.types import SpanRecord, missing_required_attributes

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "trace.yaml"
DEFAULT_DIR = PROJECT_ROOT / "data" / "traces"
DEFAULT_REDACT_KEYS: tuple[str, ...] = ("api_key", "authorization", "token", "password")

REDACTED = "***"
PREVIEW_CHARS = 80

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}

# Value-level scrubs: belt and braces for secrets that arrive inside a value
# (an exception message, a URL) instead of under a redactable key.
_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)\b(api[_-]?key|password|authorization|token|secret)\s*[:=]\s*\S+"),
)
_SEGMENT_SPLIT = re.compile(r"[^a-z0-9]+")


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TracerConfig:
    """Runtime tracer settings, resolved from config/trace.yaml + env."""

    enabled: bool = True
    directory: Path = DEFAULT_DIR
    redact_keys: tuple[str, ...] = DEFAULT_REDACT_KEYS
    include_prompt: bool = False


_state: dict[str, Any] = {"config": None, "exporter": None}
_TRACER: Optional["Tracer"] = None

_current_span: contextvars.ContextVar[Optional["Span"]] = contextvars.ContextVar(
    "contextlab_current_span", default=None
)


def _build_config() -> TracerConfig:
    """config/trace.yaml, then CONTEXTLAB_TRACE_* environment overrides."""
    raw: dict = {}
    if DEFAULT_CONFIG_PATH.exists():
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}

    enabled = bool(raw.get("enabled", True))
    directory = Path(raw.get("dir") or DEFAULT_DIR)
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    redact_keys = tuple(raw.get("redact_keys") or DEFAULT_REDACT_KEYS)
    include_prompt = bool(raw.get("include_prompt", False))

    env_enabled = os.environ.get("CONTEXTLAB_TRACE_ENABLED", "").strip().lower()
    if env_enabled in _TRUTHY:
        enabled = True
    elif env_enabled in _FALSY:
        enabled = False

    env_dir = os.environ.get("CONTEXTLAB_TRACE_DIR")
    if env_dir:
        directory = Path(env_dir).expanduser()

    env_prompt = os.environ.get("CONTEXTLAB_TRACE_INCLUDE_PROMPT", "").strip().lower()
    if env_prompt in _TRUTHY:
        include_prompt = True
    elif env_prompt in _FALSY:
        include_prompt = False

    env_keys = os.environ.get("CONTEXTLAB_TRACE_REDACT_KEYS")
    if env_keys:
        redact_keys = tuple(k.strip() for k in env_keys.split(",") if k.strip())

    return TracerConfig(
        enabled=enabled,
        directory=directory,
        redact_keys=redact_keys,
        include_prompt=include_prompt,
    )


def get_config() -> TracerConfig:
    """Cached tracer configuration."""
    if _state["config"] is None:
        _state["config"] = _build_config()
    return _state["config"]


def configure(
    enabled: Optional[bool] = None,
    directory: Optional[str | Path] = None,
    redact_keys: Optional[tuple[str, ...]] = None,
    include_prompt: Optional[bool] = None,
    exporter: Optional[SpanExporter] = None,
) -> TracerConfig:
    """Override tracer settings in-process (tests use exporter=InMemoryExporter())."""
    changes: dict[str, Any] = {}
    if enabled is not None:
        changes["enabled"] = enabled
    if directory is not None:
        changes["directory"] = Path(directory)
    if redact_keys is not None:
        changes["redact_keys"] = tuple(redact_keys)
    if include_prompt is not None:
        changes["include_prompt"] = include_prompt

    config = replace(get_config(), **changes) if changes else get_config()
    _state["config"] = config
    if exporter is not None:
        _state["exporter"] = exporter
    _invalidate_tracer()
    return config


def reset() -> None:
    """Drop cached config/exporter and re-read config/trace.yaml + env."""
    _state["config"] = None
    _state["exporter"] = None
    _invalidate_tracer()


def _invalidate_tracer() -> None:
    global _TRACER
    _TRACER = None


def get_exporter() -> SpanExporter:
    """Explicitly configured exporter, else JSONL (or nothing when disabled)."""
    if _state["exporter"] is not None:
        return _state["exporter"]
    config = get_config()
    return JSONLExporter(config.directory) if config.enabled else NullExporter()


def get_tracer() -> "Tracer":
    """The process-wide tracer, rebuilt whenever config or exporter changes."""
    global _TRACER
    if _TRACER is None:
        _TRACER = Tracer(get_config(), get_exporter())
    return _TRACER


# ── Attribute safety ──────────────────────────────────────────────────────────


def _json_safe(value: Any) -> Any:
    """Attribute values are JSON primitives or lists thereof. Nothing else."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(str(v) for v in value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _clean_attributes(attributes: Optional[dict]) -> dict:
    return {str(k): _json_safe(v) for k, v in (attributes or {}).items()}


def key_is_secret(key: str, redact_keys: tuple[str, ...]) -> bool:
    """True when an attribute key names a secret.

    Matching is case-insensitive and segment-aware: a redact key hits a whole
    `_`/`-`/`.`-separated segment of the key, or appears verbatim when it spans
    several segments ("api_key"). Token *counts* are intentionally left alone —
    `prompt_tokens` / `input_tokens` are span contracts, not credentials.
    """
    lowered = key.lower()
    segments = set(_SEGMENT_SPLIT.split(lowered))
    for redact_key in redact_keys:
        redact_key = redact_key.lower()
        if redact_key in segments:
            return True
        if _SEGMENT_SPLIT.search(redact_key) and redact_key in lowered:
            return True
    return False


def scrub_text(text: str) -> str:
    """Replace credential-looking substrings inside a value."""
    text = _SECRET_VALUE_PATTERNS[0].sub(REDACTED, text)
    text = _SECRET_VALUE_PATTERNS[1].sub(f"Bearer {REDACTED}", text)
    text = _SECRET_VALUE_PATTERNS[2].sub(lambda m: f"{m.group(1)}={REDACTED}", text)
    return text


def _scrub_value(value: Any) -> Any:
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, list):
        return [_scrub_value(v) for v in value]
    return value


def redact_record(record: SpanRecord, redact_keys: tuple[str, ...]) -> SpanRecord:
    """Copy of a record with secret-named attributes replaced by '***'."""
    safe = record.model_copy(deep=True)
    safe.attributes = {
        key: (REDACTED if key_is_secret(key, redact_keys) else _scrub_value(value))
        for key, value in record.attributes.items()
    }
    return safe


def prompt_attributes(prompt: str) -> dict:
    """Reconstructability without dumping prompts: hash + size + 80-char preview.

    The full prompt is only stored when config/trace.yaml sets
    include_prompt: true.
    """
    attributes: dict[str, Any] = {
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        "prompt_chars": len(prompt),
        "prompt_preview": " ".join(prompt.split())[:PREVIEW_CHARS],
    }
    if get_config().include_prompt:
        attributes["prompt"] = prompt
    return attributes


# ── Spans ─────────────────────────────────────────────────────────────────────


class Span:
    """A live span. Usable as a context manager; nesting is by contextvar."""

    def __init__(self, tracer: "Tracer", record: SpanRecord) -> None:
        self._tracer = tracer
        self.record = record
        self._token: Optional[contextvars.Token] = None
        self._finished = False

    # identity
    @property
    def name(self) -> str:
        return self.record.name

    @property
    def trace_id(self) -> str:
        return self.record.trace_id

    @property
    def span_id(self) -> str:
        return self.record.span_id

    @property
    def parent_span_id(self) -> Optional[str]:
        return self.record.parent_span_id

    @property
    def attributes(self) -> dict:
        return self.record.attributes

    @property
    def status(self) -> str:
        return self.record.status

    @property
    def finished(self) -> bool:
        return self._finished

    # attributes
    def set_attributes(self, **attributes: Any) -> "Span":
        """Merge attributes; values are coerced to JSON-safe primitives."""
        for key, value in attributes.items():
            self.record.attributes[str(key)] = _json_safe(value)
        return self

    def set_attribute(self, key: str, value: Any) -> "Span":
        return self.set_attributes(**{key: value})

    # lifecycle
    def fail(self, exc: BaseException) -> "Span":
        """Mark the span as failed with the exception's type and message."""
        self.record.status = "error"
        self.record.error = f"{type(exc).__name__}: {exc}"
        return self

    def finish(self, error: Optional[BaseException | str] = None) -> "Span":
        """Close the span. Idempotent, so a finalizer can call it too."""
        if self._finished:
            return self
        if isinstance(error, BaseException):
            self.fail(error)
        elif error is not None:
            self.record.status = "error"
            self.record.error = str(error)
        self.record.end_ns = time.time_ns()
        missing = missing_required_attributes(self.record)
        if missing:
            # Explicit, not silent: the hop ran but its contract is incomplete.
            self.record.attributes.setdefault("missing_context", missing)
        self._finished = True
        return self

    def close_if_open(self, error: Optional[BaseException] = None) -> "Span":
        """Close a span the owner forgot — an open span at trace end is an error."""
        if self._finished:
            return self
        if error is not None:
            self.fail(error)
        else:
            self.record.status = "error"
            self.record.error = self.record.error or "span left open when the trace finished"
        return self.finish()

    def __enter__(self) -> "Span":
        self._token = _current_span.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # First error wins: an explicit fail() (e.g. BudgetError) stays the
        # recorded reason instead of being overwritten by the propagating
        # SystemExit/Exception.
        if exc is not None and self.record.status != "error":
            self.fail(exc)
        self.finish()
        if self._token is not None:
            _current_span.reset(self._token)
            self._token = None
        return False


class _NullSpan:
    """No-op span for untraced call paths (and the disabled tracer)."""

    name = "null"
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    status = "ok"
    finished = True
    attributes: dict = {}

    def set_attributes(self, **attributes: Any) -> "_NullSpan":
        return self

    def set_attribute(self, key: str, value: Any) -> "_NullSpan":
        return self

    def fail(self, exc: BaseException) -> "_NullSpan":
        return self

    def finish(self, error: Any = None) -> "_NullSpan":
        return self

    def close_if_open(self, error: Any = None) -> "_NullSpan":
        return self

    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


_NULL_SPAN = _NullSpan()


class Trace:
    """One trace: a root span plus everything started underneath it."""

    def __init__(self, tracer: "Tracer", trace_id: str, root: Span) -> None:
        self._tracer = tracer
        self.trace_id = trace_id
        self.root = root
        self._finished = False

    @property
    def root_name(self) -> str:
        return self.root.name

    def set_attributes(self, **attributes: Any) -> "Trace":
        self.root.set_attributes(**attributes)
        return self

    def finish(self, error: Any = None) -> "Trace":
        """Close outstanding spans and hand the trace to the exporter. Idempotent."""
        if self._finished:
            return self
        self._finished = True
        self._tracer.finish_trace(self, error=error)
        return self

    def __enter__(self) -> "Trace":
        self.root.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.root.__exit__(exc_type, exc, tb)
        self.finish(exc)
        return False


class _NullTrace:
    """Returned when tracing is disabled — API-compatible, records nothing."""

    def __init__(self, name: str) -> None:
        self.trace_id: Optional[str] = None
        self.root_name = name
        self.root = _NULL_SPAN

    def set_attributes(self, **attributes: Any) -> "_NullTrace":
        return self

    def finish(self, error: Any = None) -> "_NullTrace":
        return self

    def __enter__(self) -> "_NullTrace":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class Tracer:
    """Creates spans, tracks parentage, exports finished traces."""

    def __init__(self, config: TracerConfig, exporter: SpanExporter) -> None:
        self.config = config
        self.exporter = exporter
        self._spans: dict[str, list[Span]] = {}

    def start_trace(self, name: str, attributes: Optional[dict] = None) -> Trace:
        """Open a new root trace (32 hex trace id) and its root span."""
        record = SpanRecord(
            trace_id=secrets.token_hex(16),
            span_id=secrets.token_hex(8),
            parent_span_id=None,
            name=name,
            start_ns=time.time_ns(),
            attributes=_clean_attributes(attributes),
        )
        root = Span(self, record)
        self._spans[record.trace_id] = [root]
        return Trace(self, record.trace_id, root)

    def start_span(self, name: str, parent: Span, attributes: Optional[dict] = None) -> Span:
        """Open a span under `parent` (16 hex span id)."""
        record = SpanRecord(
            trace_id=parent.trace_id,
            span_id=secrets.token_hex(8),
            parent_span_id=parent.span_id,
            name=name,
            start_ns=time.time_ns(),
            attributes=_clean_attributes(attributes),
        )
        span = Span(self, record)
        self._spans.setdefault(record.trace_id, []).append(span)
        return span

    def spans_for(self, trace_id: str) -> list[Span]:
        """Live spans of a trace, in creation order."""
        return list(self._spans.get(trace_id, []))

    def finish_trace(self, trace: Trace, error: Any = None) -> None:
        """Close leftovers (as errors), redact, and export a snapshot of the trace."""
        spans = self._spans.pop(trace.trace_id, [])
        for span in spans:
            span.close_if_open(error if isinstance(error, BaseException) else None)

        records = [
            redact_record(span.record.model_copy(deep=True), self.config.redact_keys)
            for span in spans
        ]
        self.exporter.export(records)


# ── Module-level API ──────────────────────────────────────────────────────────


def start_trace(name: str, **attributes: Any) -> Trace:
    """Start a root trace. Returns a no-op Trace when tracing is disabled."""
    if not get_config().enabled:
        return _NullTrace(name)
    return get_tracer().start_trace(name, attributes)


def start_span(name: str, parent: Optional[Span] = None, **attributes: Any) -> Span:
    """Start a span under `parent`, or under the current span of this context."""
    span_parent = parent if parent is not None else _current_span.get()
    if span_parent is None:
        return _NULL_SPAN
    return get_tracer().start_span(name, span_parent, attributes)


def current_span() -> Optional[Span]:
    """The span this context is inside, or None."""
    return _current_span.get()


def current_trace_id() -> Optional[str]:
    """Trace id of the current context, or None when untraced."""
    span = _current_span.get()
    return span.trace_id if span is not None else None

"""Slice 4 tracer tests: parentage, timing, redaction, missing-context attributes.

They assert the hop contracts from the brief:
  retrieve.*    query, mode, k, chunk_ids, doc_ids
  assemble.pack budget_tokens, prompt_tokens, kept_ids, dropped_ids
  generate      model, input_tokens, output_tokens, latency_ms
  eval.case     case_id, suite, passed
"""
import json
import re
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from contextlab.assemble import assemble
from contextlab.config import get_config
from contextlab.evals.report import write_report
from contextlab.evals.types import CaseResult, EvalReport
from contextlab.retrieve import Retriever, ordered_unique
from contextlab.trace import (
    InMemoryExporter,
    REQUIRED_ATTRIBUTES,
    build_summary,
    configure,
    key_is_secret,
    load_records,
    missing_required_attributes,
    prompt_attributes,
    reset,
    start_span,
    start_trace,
)
from contextlab.trace.show import render_trace, render_trace_list, select_trace_id, spans_of
from contextlab.types import AssembleRequest, RetrieveRequest

TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
SPAN_ID_RE = re.compile(r"^[0-9a-f]{16}$")


@pytest.fixture
def exporter():
    """In-memory exporter — assertions never touch the repo's trace files."""
    memory = InMemoryExporter()
    configure(exporter=memory)
    yield memory
    reset()


def _retriever() -> Retriever:
    """Retriever built from config/retrieval.yaml (same as the CLI path)."""
    config = get_config()
    return Retriever(
        chunk_tokens=config["chunk_tokens"],
        chunk_overlap_tokens=config["chunk_overlap_tokens"],
        encoding=config["encoding"],
        embedding_model=config["embedding_model"],
        bm25_top_n=config["bm25_top_n"],
        dense_top_n=config["dense_top_n"],
        rrf_k=config["rrf_k"],
        hybrid_k=config["hybrid_k"],
    )


def _by_name(records) -> dict:
    return {record.name: record for record in records}


# ── ids and parentage ─────────────────────────────────────────────────────────


def test_trace_and_span_id_formats(exporter):
    """trace_id is 32 lowercase hex, span_id 16 — the OTel shapes."""
    with start_trace("assemble") as trace:
        with start_span("retrieve.hybrid") as span:
            pass

    assert TRACE_ID_RE.match(trace.trace_id)
    assert SPAN_ID_RE.match(span.span_id)
    assert SPAN_ID_RE.match(trace.root.span_id)
    assert len({r.span_id for r in exporter.records}) == len(exporter.records)


def test_nested_spans_get_parent_span_id(exporter):
    """Parentage is recorded, not inferred: every child points at its span."""
    with start_trace("assemble") as trace:
        with start_span("retrieve.hybrid") as hybrid:
            with start_span("retrieve.bm25") as bm25:
                pass
            with start_span("retrieve.fuse") as fuse:
                pass

    records = _by_name(exporter.for_trace(trace.trace_id))
    assert records["assemble"].parent_span_id is None
    assert records["retrieve.hybrid"].parent_span_id == trace.root.span_id
    assert records["retrieve.bm25"].parent_span_id == hybrid.span_id
    assert records["retrieve.fuse"].parent_span_id == hybrid.span_id
    assert bm25.span_id != fuse.span_id

    # every span of the trace shares one trace_id
    assert {r.trace_id for r in exporter.records} == {trace.trace_id}


def test_parentage_reconstructed_from_jsonl(tmp_path):
    """A unit test can rebuild the parent/child tree from the JSONL alone."""
    configure(directory=tmp_path)
    with start_trace("assemble") as trace:
        with start_span("assemble.pack", budget_tokens=800) as pack:
            pass

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    assert files[0].name == f"{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"

    lines = files[0].read_text().strip().splitlines()
    assert len(lines) == 2  # one JSON object per span per line
    parsed = [json.loads(line) for line in lines]
    assert {row["name"] for row in parsed} == {"assemble", "assemble.pack"}
    assert parsed[1]["parent_span_id"] == parsed[0]["span_id"] == trace.root.span_id

    records = load_records(tmp_path)
    assert _by_name(records)["assemble.pack"].parent_span_id == trace.root.span_id
    assert all(SPAN_ID_RE.match(r.span_id) and TRACE_ID_RE.match(r.trace_id) for r in records)
    assert pack.record.end_ns is not None


# ── timing ────────────────────────────────────────────────────────────────────


def test_span_timing_brackets_the_work(exporter):
    """end_ns >= start_ns, durations are plausible, children fit inside parents."""
    with start_trace("assemble") as trace:
        with start_span("retrieve.hybrid") as hybrid:
            time.sleep(0.02)

    records = _by_name(exporter.for_trace(trace.trace_id))
    root, child = records["assemble"], records["retrieve.hybrid"]

    assert root.end_ns >= root.start_ns
    assert child.end_ns >= child.start_ns
    assert child.duration_ms >= 15  # slept 20 ms
    assert root.duration_ms >= child.duration_ms

    summary = build_summary(list(exporter.for_trace(trace.trace_id)))
    assert summary.span_count == 2
    assert summary.duration_ms == pytest.approx(root.duration_ms, rel=1e-6)
    assert summary.root_name == "assemble"


# ── redaction ─────────────────────────────────────────────────────────────────


def test_redact_keys_match_segments_not_token_counts():
    """Secret keys are redacted; prompt_tokens/input_tokens stay readable."""
    keys = ("api_key", "authorization", "token", "password")
    assert key_is_secret("api_key", keys)
    assert key_is_secret("OPENAI_API_KEY", keys)
    assert key_is_secret("HTTP_Authorization", keys)
    assert key_is_secret("db_password", keys)
    assert key_is_secret("token", keys)
    assert not key_is_secret("prompt_tokens", keys)
    assert not key_is_secret("input_tokens", keys)
    assert not key_is_secret("output_tokens", keys)
    assert not key_is_secret("kept_ids", keys)


def test_secret_attributes_are_redacted_on_export(exporter):
    """A span carrying api_key/authorization lands redacted in the export."""
    with start_trace("assemble") as trace:
        span = start_span("generate", model="gpt-4o-mini")
        span.set_attributes(
            api_key="sk-live-abcdefghijklmnop",
            Authorization="Bearer abcdefghijklmnop",
            user_password="hunter2",
            token="t-1234567890",
            note="token: oops-a-secret-1234",
            input_tokens=11,
            output_tokens=3,
            latency_ms=12.5,
        )
        span.finish()

    record = _by_name(exporter.for_trace(trace.trace_id))["generate"]
    assert record.attributes["api_key"] == "***"
    assert record.attributes["Authorization"] == "***"
    assert record.attributes["user_password"] == "***"
    assert record.attributes["token"] == "***"
    assert "oops-a-secret-1234" not in record.attributes["note"]
    # token counts are span contracts, not credentials
    assert record.attributes["input_tokens"] == 11
    assert record.attributes["output_tokens"] == 3
    # and the span still satisfies the generate hop contract
    assert missing_required_attributes(record) == []


def test_jsonl_never_contains_raw_credentials(tmp_path):
    """No API keys in data/traces/*.jsonl — only the redacted marker."""
    configure(directory=tmp_path)
    with start_trace("assemble"):
        span = start_span("generate", model="gpt-4o-mini", api_key="sk-abcdefghijklmnop")
        span.set_attributes(input_tokens=1, output_tokens=1, latency_ms=1.0)
        span.finish()

    blob = "".join(path.read_text() for path in tmp_path.glob("*.jsonl"))
    assert "sk-abcdefghijklmnop" not in blob
    assert "***" in blob


# ── missing-context attributes ────────────────────────────────────────────────


def test_incomplete_hop_is_marked_missing_context(exporter):
    """A hop that closes without its contract records what it is missing."""
    with start_trace("assemble") as trace:
        span = start_span("retrieve.hybrid", query="E-4471")
        span.finish()

    record = _by_name(exporter.for_trace(trace.trace_id))["retrieve.hybrid"]
    assert record.attributes["missing_context"] == ["mode", "k", "chunk_ids", "doc_ids"]
    assert missing_required_attributes(record) == [
        "mode", "k", "chunk_ids", "doc_ids",
    ]


def test_complete_hops_carry_no_missing_marker(exporter):
    """The real retrieve/assemble hops satisfy their attribute contracts."""
    retriever = _retriever()
    with start_trace("retrieve", query="E-4471") as retrieve_trace:
        retriever.retrieve(RetrieveRequest(query="E-4471", k=5, mode="hybrid"))

    with start_trace("assemble", query="what does E-4471 mean") as assemble_trace:
        assemble(AssembleRequest(
            query="what does E-4471 mean",
            system="You are a helpful assistant.",
            budget_tokens=800,
            retrieve=False,
        ))

    for trace_id in (retrieve_trace.trace_id, assemble_trace.trace_id):
        records = exporter.for_trace(trace_id)
        assert records
        for record in records:
            assert "missing_context" not in record.attributes, record.name
            assert missing_required_attributes(record) == []

    # retrieve=False means no retrieval hop at all
    assert not [
        r for r in exporter.for_trace(assemble_trace.trace_id) if r.name.startswith("retrieve.")
    ]


# ── retrieve hop ──────────────────────────────────────────────────────────────


def test_retrieve_span_chunk_ids_match_the_hit_list(exporter):
    """chunk_ids on the retrieve span equals the Hit list the caller got back."""
    retriever = _retriever()
    with start_trace("retrieve", query="E-4471", mode="hybrid", k=5) as trace:
        response = retriever.retrieve(RetrieveRequest(query="E-4471", k=5, mode="hybrid"))

    records = _by_name(exporter.for_trace(trace.trace_id))
    hybrid = records["retrieve.hybrid"]
    expected_ids = [hit.chunk_id for hit in response.hits]

    assert hybrid.attributes["chunk_ids"] == expected_ids
    assert hybrid.attributes["doc_ids"] == ordered_unique(hit.doc_id for hit in response.hits)
    assert hybrid.attributes["query"] == "E-4471"
    assert hybrid.attributes["mode"] == "hybrid"
    assert hybrid.attributes["k"] == 5

    # the sub-hops hang off retrieve.hybrid, and bm25/dense were asked for top_n
    assert records["retrieve.bm25"].parent_span_id == hybrid.span_id
    assert records["retrieve.dense"].parent_span_id == hybrid.span_id
    assert records["retrieve.fuse"].parent_span_id == hybrid.span_id
    assert records["retrieve.bm25"].attributes["k"] == get_config()["bm25_top_n"]
    assert records["retrieve.dense"].attributes["k"] == get_config()["dense_top_n"]
    assert records["retrieve.fuse"].attributes["chunk_ids"] == expected_ids

    # trace_id is wired onto the response settings
    assert response.settings["trace_id"] == trace.trace_id


def test_retrieve_error_still_exports_the_trace(exporter, monkeypatch):
    """A failing retriever leaves an error span and a written trace."""
    retriever = _retriever()

    def boom(*args, **kwargs):
        raise RuntimeError("bm25 exploded")

    monkeypatch.setattr(retriever.bm25, "query", boom)

    with pytest.raises(RuntimeError):
        with start_trace("retrieve", query="E-4471") as trace:
            retriever.retrieve(RetrieveRequest(query="E-4471", k=5, mode="hybrid"))

    records = exporter.for_trace(trace.trace_id)
    assert records, "trace must still be exported when a hop raises"
    errored = _by_name(records)
    assert errored["retrieve.bm25"].status == "error"
    assert "RuntimeError: bm25 exploded" in errored["retrieve.bm25"].error
    assert errored["retrieve.hybrid"].status == "error"
    assert errored["retrieve.hybrid"].error.startswith("RuntimeError")


# ── assemble hop ──────────────────────────────────────────────────────────────


def test_assemble_pack_span_lists_kept_and_dropped_ids(exporter):
    """A tight budget produces dropped_ids on the pack span."""
    with start_trace("assemble", query="what does E-4471 mean") as trace:
        ctx = assemble(AssembleRequest(
            query="what does E-4471 mean",
            system="You are a helpful assistant.",
            budget_tokens=250,
            retrieve=True,
            retrieve_k=8,
            retrieve_mode="hybrid",
        ))

    records = _by_name(exporter.for_trace(trace.trace_id))
    pack = records["assemble.pack"]
    assert pack.attributes["budget_tokens"] == 250
    assert pack.attributes["prompt_tokens"] == ctx.prompt_tokens
    assert pack.attributes["kept_ids"] == [
        block.ref_id for block in ctx.blocks if block.kind in ("retrieval", "memory", "tool")
    ]
    assert pack.attributes["dropped_ids"] == [block.ref_id for block in ctx.dropped]
    assert pack.attributes["dropped_ids"], "a 250-token budget must drop retrieval blocks"
    assert ctx.settings["trace_id"] == trace.trace_id

    # the retrieve hop is a sibling of the pack hop, not a child of it
    assert records["retrieve.hybrid"].parent_span_id == trace.root.span_id
    assert pack.parent_span_id == trace.root.span_id


def test_prompt_is_hashed_not_stored(exporter):
    """include_prompt defaults to false: hash + size + 80-char preview only."""
    prompt = "x" * 400
    with start_trace("assemble") as trace:
        span = start_span("assemble.pack", budget_tokens=800)
        span.set_attributes(
            prompt_tokens=100, kept_ids=[], dropped_ids=[], **prompt_attributes(prompt)
        )
        span.finish()

    attributes = _by_name(exporter.for_trace(trace.trace_id))["assemble.pack"].attributes
    assert "prompt" not in attributes
    assert attributes["prompt_chars"] == 400
    assert len(attributes["prompt_preview"]) == 80
    assert len(attributes["prompt_sha256"]) == 16
    assert missing_required_attributes(
        _by_name(exporter.for_trace(trace.trace_id))["assemble.pack"]
    ) == []


def test_include_prompt_opt_in_writes_the_full_prompt(tmp_path):
    """include_prompt: true is the only way a full prompt reaches a span."""
    configure(directory=tmp_path, include_prompt=True)
    with start_trace("assemble"):
        span = start_span("assemble.pack", budget_tokens=800)
        span.set_attributes(prompt_tokens=2, kept_ids=[], dropped_ids=[], **prompt_attributes("full prompt text"))
        span.finish()

    records = load_records(tmp_path)
    assert _by_name(records)["assemble.pack"].attributes["prompt"] == "full prompt text"


# ── generate hop ──────────────────────────────────────────────────────────────


class _FakeUsage:
    prompt_tokens = 11
    completion_tokens = 3


class _FakeCompletion:
    usage = _FakeUsage()
    choices = [SimpleNamespace(message=SimpleNamespace(content="hello"))]


def _fake_openai_module():
    class _Completions:
        @staticmethod
        def create(**kwargs):
            return _FakeCompletion()

    class _Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=_Completions())

    return SimpleNamespace(OpenAI=_Client)


def test_generate_hop_span_records_model_and_tokens(exporter, monkeypatch):
    """generate carries model / input_tokens / output_tokens / latency_ms."""
    from contextlab.evals import suites_answer

    monkeypatch.setitem(sys.modules, "openai", _fake_openai_module())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key-0000")

    with start_trace("eval.case", case_id="ans002", suite="answer"):
        output, latency_ms, cost_usd = suites_answer.generate_answer("q", "some assembled prompt")

    generate = _by_name(exporter.records)["generate"]
    assert output == "hello"
    assert generate.attributes["model"] == "gpt-4o-mini"
    assert generate.attributes["input_tokens"] == 11
    assert generate.attributes["output_tokens"] == 3
    assert generate.attributes["latency_ms"] >= 0
    assert missing_required_attributes(generate) == []
    assert "prompt" not in generate.attributes
    assert generate.attributes["prompt_preview"] == "some assembled prompt"
    assert cost_usd > 0


# ── eval.case hop ─────────────────────────────────────────────────────────────


def test_eval_case_spans_and_trace_id(exporter, tmp_path):
    """Suite cases get eval.case spans and CaseResult.trace_id."""
    from contextlab.evals import suites_assembly

    golden = tmp_path / "assembly_one.jsonl"
    golden.write_text(json.dumps({
        "id": "a900",
        "suite": "assembly",
        "query": "What is the current refund window?",
        "budget_tokens": 600,
        "intent": "lexical",
    }) + "\n")

    results, metrics = suites_assembly.run_suite(golden_path=golden)

    assert metrics.n == 1
    assert len(results) == 1
    case_result = results[0]
    assert case_result.trace_id and TRACE_ID_RE.match(case_result.trace_id)

    case_span = _by_name(exporter.for_trace(case_result.trace_id))["eval.case"]
    assert case_span.parent_span_id is None
    assert case_span.attributes["case_id"] == "a900"
    assert case_span.attributes["suite"] == "assembly"
    assert case_span.attributes["passed"] == case_result.passed
    assert missing_required_attributes(case_span) == []


def test_eval_report_serializes_trace_id(tmp_path):
    """artifacts/eval_report.json carries trace_id per case."""
    report = EvalReport(
        started_at="2026-09-14T00:00:00Z",
        finished_at="2026-09-14T00:01:00Z",
        suites={},
        gates=[],
        cases=[
            CaseResult(id="r001", suite="retrieval", passed=True, trace_id="a" * 32),
            CaseResult(id="a001", suite="assembly", passed=False, trace_id="b" * 32),
        ],
        passed=True,
    )

    path = tmp_path / "eval_report.json"
    write_report(report, path)
    data = json.loads(path.read_text())

    assert data["cases"][0]["trace_id"] == "a" * 32
    assert data["cases"][1]["trace_id"] == "b" * 32
    assert all(case["trace_id"] for case in data["cases"])


# ── show CLI surface ──────────────────────────────────────────────────────────


def test_show_last_prints_the_tree_with_chunk_ids(tmp_path, capsys):
    """`trace show --last` renders the tree, the ids retrieve returned, drops."""
    configure(directory=tmp_path)
    with start_trace("assemble", query="what does E-4471 mean") as trace:
        with start_span("retrieve.hybrid", query="what does E-4471 mean", mode="hybrid", k=8) as hybrid:
            hybrid.set_attributes(
                chunk_ids=["error_codes::c0001", "incident_runbook::c0000"],
                doc_ids=["error_codes", "incident_runbook"],
            )
        pack = start_span("assemble.pack", budget_tokens=400)
        pack.set_attributes(
            prompt_tokens=380,
            kept_ids=["error_codes::c0001"],
            dropped_ids=["incident_runbook::c0000"],
        )
        pack.finish()

    records = load_records(tmp_path)
    assert select_trace_id(records, last=True) == trace.trace_id
    assert select_trace_id(records, trace_id=trace.trace_id) == trace.trace_id
    assert select_trace_id(records, trace_id="0" * 32) is None

    from contextlab.trace.__main__ import main

    assert main(["show", "--last", "--dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out

    assert trace.trace_id in out
    assert "chunk_ids    [2] error_codes::c0001, incident_runbook::c0000" in out
    assert "dropped_ids  [1] incident_runbook::c0000" in out
    assert "\n  retrieve.hybrid" in out  # indented => child of the root
    assert "\n  assemble.pack" in out
    assert "span tree" in out

    # `--case` finds the trace through its eval.case span
    with start_trace("eval.case", case_id="r001", suite="retrieval") as case_trace:
        case_trace.set_attributes(passed=True)
    records = load_records(tmp_path)
    assert select_trace_id(records, case_id="r001") == case_trace.trace_id
    assert select_trace_id(records, last=True) == case_trace.trace_id
    assert main(["show", "--case", "r001", "--dir", str(tmp_path)]) == 0
    assert "eval.case" in capsys.readouterr().out


def test_show_reports_missing_trace_and_empty_dir(tmp_path, capsys):
    """Unknown ids fail loudly instead of printing an empty tree."""
    from contextlab.trace.__main__ import main

    assert main(["show", "--last", "--dir", str(tmp_path / "nope")]) == 1
    assert "no spans" in capsys.readouterr().err

    with start_trace("assemble"):
        pass

    assert main(["show", "--case", "r999", "--dir", str(tmp_path / "traces")]) == 1
    assert "no trace found" in capsys.readouterr().err


def test_render_trace_list_reports_size(tmp_path):
    """`trace list` shows trace id, root, span count and duration."""
    configure(directory=tmp_path)
    with start_trace("ingest") as trace:
        with start_span("retrieve.bm25", query="q") as span:
            span.set_attributes(mode="bm25", k=1, chunk_ids=[], doc_ids=[])

    rendered = render_trace_list(load_records(tmp_path))
    assert trace.trace_id in rendered
    assert "ingest" in rendered
    assert "spans" in rendered


def test_trace_tree_marks_missing_context(exporter):
    """`show` surfaces hops that closed without their contract."""
    with start_trace("assemble") as trace:
        start_span("retrieve.hybrid", query="E-4471").finish()

    rendered = render_trace(spans_of(exporter.records, trace.trace_id))
    assert "MISSING" in rendered
    assert "chunk_ids" in rendered


# ── disabled tracing ──────────────────────────────────────────────────────────


def test_tracer_disabled_records_nothing(exporter):
    """enabled: false means no spans, no ids, no writes — and no crash."""
    configure(enabled=False)
    with start_trace("assemble") as trace:
        with start_span("retrieve.hybrid", query="E-4471") as span:
            span.set_attributes(mode="hybrid", k=5, chunk_ids=[], doc_ids=[])

    assert trace.trace_id is None
    assert span.trace_id is None
    assert exporter.records == []


def test_untraced_library_calls_record_nothing(exporter):
    """retrieve/assemble called outside a trace stay silent (library default)."""
    retriever = _retriever()
    response = retriever.retrieve(RetrieveRequest(query="E-4471", k=5, mode="bm25"))
    ctx = assemble(AssembleRequest(
        query="q", system="You are a helpful assistant.", budget_tokens=800, retrieve=False
    ))

    assert exporter.records == []
    assert "trace_id" not in response.settings
    assert "trace_id" not in ctx.settings


def test_required_attribute_table_matches_the_brief():
    """The hop contracts are declared, not implied."""
    assert REQUIRED_ATTRIBUTES["retrieve.*"] == ("query", "mode", "k", "chunk_ids", "doc_ids")
    assert REQUIRED_ATTRIBUTES["assemble.pack"] == (
        "budget_tokens", "prompt_tokens", "kept_ids", "dropped_ids",
    )
    assert REQUIRED_ATTRIBUTES["generate"] == (
        "model", "input_tokens", "output_tokens", "latency_ms",
    )
    assert REQUIRED_ATTRIBUTES["eval.case"] == ("case_id", "suite", "passed")


def test_redaction_survives_the_in_memory_exporter():
    """The exporter receives already-redacted copies (see tracer.finish_trace)."""
    memory = InMemoryExporter()
    configure(exporter=memory)
    configure(exporter=memory, redact_keys=("api_key", "authorization", "token", "password"))
    with start_trace("assemble"):
        start_span("generate", model="m", api_key="sk-zzzzzzzzzzzzzzzz").finish()

    assert _by_name(memory.records)["generate"].attributes["api_key"] == "***"

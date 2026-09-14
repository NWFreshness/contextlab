"""Slice 5 router tests: signals, policy, fallback, cost, goldens, tracing.

No test needs a network or an API key: the eval path is dry-run by definition,
and the generate wrapper is exercised with a stub client shaped like the SDK.
"""
import json
import re
from collections import Counter

import pytest
import yaml

from contextlab.assemble import assemble
from contextlab.router import (
    GenerateRequest,
    RouterConfig,
    RouterConfigError,
    assemble_meta,
    decide,
    estimate_cost_usd,
    extract_signals,
    generate,
    has_identifier,
    load_router_config,
    resolve_model,
    route_request,
)
from contextlab.trace import InMemoryExporter, configure, reset, start_trace
from contextlab.types import AssembleRequest

SHIPPED_MODELS = {"cheap": "test-cheap", "strong": "test-strong", "fallback": "test-fallback"}


@pytest.fixture
def exporter():
    """In-memory exporter — span assertions never touch data/traces/."""
    memory = InMemoryExporter()
    configure(exporter=memory)
    yield memory
    reset()


def config_for(**overrides) -> RouterConfig:
    """Shipped rules/keywords, but with model names a stub client can serve."""
    data = load_router_config().model_dump()
    data["models"] = dict(SHIPPED_MODELS)
    data["prices_usd_per_1m"] = {name: None for name in SHIPPED_MODELS.values()}
    data.update(overrides)
    return RouterConfig(**data)


def write_config(tmp_path, **overrides):
    """A config file on disk for the load/validate tests."""
    from contextlab.router.policy import DEFAULT_ROUTER_CONFIG

    raw = yaml.safe_load(DEFAULT_ROUTER_CONFIG.read_text(encoding="utf-8"))
    raw.update(overrides)
    path = tmp_path / "router.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def stub_client(fail_models=(), output="stub output", usage=True):
    """A client shaped like the OpenAI SDK, recording the models it was asked for."""
    class _Completions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["model"] in fail_models:
                raise RuntimeError(f"{kwargs['model']} is down")
            usage_block = (
                type("Usage", (), {"prompt_tokens": 12, "completion_tokens": 5})() if usage else None
            )
            return type("Response", (), {
                "usage": usage_block,
                "choices": [type("Choice", (), {"message": type("Message", (), {"content": output})()})()],
            })()

    completions = _Completions()
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    client.calls = completions.calls
    return client


# ── signals ───────────────────────────────────────────────────────────────────


def test_identifier_detection_accepts_codes_and_rejects_noise():
    assert has_identifier("What does E-4471 mean?")
    assert has_identifier("What is POL-REF-30?")
    assert has_identifier("explain e-1001")            # case-insensitive
    assert not has_identifier("Who is on call this weekend?")
    assert not has_identifier("Is there an on-call rotation?")
    assert not has_identifier("What is the 1-800 support number?")
    assert not has_identifier("How long does West Coast shipping take?")


def test_intent_keywords_follow_the_configured_order():
    config = load_router_config()
    keywords, order = config.intents["keywords"], config.intents["order"]

    def intent(query, **meta):
        return extract_signals(query, meta, config=config).intent_hint

    assert intent("What was the old refund window?") == "version"
    assert intent("Which refund policy is current?") == "version"
    assert intent("What is the SLA for ground shipping?") == "table"
    assert intent("How many users can use the Professional plan?") == "table"
    assert intent("How do I send this back if it does not fit?") == "paraphrase"
    assert intent("What is the weather in Vancouver tomorrow?") == "negative"
    assert intent("Translate the refund policy into French") == "negative"
    assert intent("What does E-4471 mean?") == "lexical"
    assert intent("Summarize the onboarding FAQ") == "other"
    # word-bounded matching: "rate" must not fire inside "corporate"
    assert intent("What is the corporate travel policy?") == "other"
    assert keywords["table"], "table keywords must come from config/router.yaml"
    assert order[0] == "negative", "negative is checked first on purpose"


def test_negative_on_empty_retrieval_but_not_when_retrieval_never_ran():
    config = load_router_config()
    empty = extract_signals(
        "quarterly revenue guidance", {"n_retrieved": 0, "retrieval_ran": True}, config=config
    )
    assert empty.intent_hint == "negative"

    not_run = extract_signals(
        "quarterly revenue guidance", {"n_retrieved": 0, "retrieval_ran": False}, config=config
    )
    assert not_run.intent_hint == "other", "0 hits without a retrieval must not read as negative"


def test_conflict_needs_both_memory_and_retrieved_hits():
    config = load_router_config()

    both = extract_signals(
        "What is our support response time?",
        {"n_retrieved": 3, "retrieval_ran": True, "memory_count": 1},
        config=config,
    )
    assert both.conflict is True

    memory_only = extract_signals(
        "What is our support response time?",
        {"n_retrieved": 0, "retrieval_ran": True, "memory_count": 2},
        config=config,
    )
    assert memory_only.conflict is False

    hits_only = extract_signals(
        "What is our support response time?",
        {"n_retrieved": 3, "retrieval_ran": True, "memory_count": 0},
        config=config,
    )
    assert hits_only.conflict is False


def test_assemble_meta_reads_the_assemble_step():
    ctx = assemble(AssembleRequest(
        query="what does E-4471 mean",
        system="You are a helpful assistant.",
        budget_tokens=800,
        retrieve=False,
    ))
    meta = assemble_meta(ctx)
    assert meta["n_retrieved"] == 0
    assert meta["retrieval_ran"] is False
    assert meta["budget_tokens"] == 800
    assert assemble_meta(ctx, memory_count=2)["memory_count"] == 2


# ── policy ────────────────────────────────────────────────────────────────────


def test_identifier_routes_cheap_and_the_reason_names_the_signal():
    config = load_router_config()
    decision = decide(extract_signals("What does E-4471 mean?", config=config), config=config)

    assert decision.route == "cheap"
    assert decision.model == config.models["cheap"]
    assert "identifier" in decision.reason
    assert decision.reason.startswith("rule[2]")
    assert decision.signals.has_identifier is True


def test_version_query_routes_strong():
    config = load_router_config()
    decision = decide(
        extract_signals("What was the refund window before we changed it?", config=config), config=config
    )
    assert decision.route == "strong"
    assert decision.model == config.models["strong"]


def test_negative_query_routes_cheap():
    config = load_router_config()
    decision = decide(extract_signals("What is the weather in Vancouver?", config=config), config=config)
    assert decision.route == "cheap"
    assert decision.reason.startswith("rule[1]")


def test_table_and_paraphrase_queries_route_strong():
    config = load_router_config()
    for query in (
        "What is the SLA for ground shipping?",
        "How many users can use the Professional plan?",
        "How do I send this back if it does not fit?",
        "How long do I have to send something back for a refund?",
    ):
        decision = decide(extract_signals(query, config=config), config=config)
        assert decision.route == "strong", f"{query} -> {decision.reason}"


def test_conflict_routes_strong_without_any_keyword_intent():
    config = load_router_config()
    signals = extract_signals(
        "What is our support response time?",
        {"n_retrieved": 5, "retrieval_ran": True, "memory_count": 1},
        config=config,
    )
    assert signals.intent_hint == "other", "only the conflict signal may fire here"
    assert signals.has_identifier is False

    decision = decide(signals, config=config)
    assert decision.route == "strong"
    assert "conflict" in decision.reason
    assert decision.reason.startswith("rule[0]")


def test_unmatched_query_falls_back_to_the_default_route():
    config = load_router_config()
    decision = decide(extract_signals("Summarize the onboarding FAQ", config=config), config=config)
    assert decision.route == config.default_route == "cheap"
    assert decision.reason.startswith("default_route=")


def test_decisions_are_deterministic():
    config = load_router_config()
    signals = extract_signals("How long do I have to send something back?", config=config)
    first = decide(signals, config=config)
    second = decide(signals, config=config)
    assert (first.route, first.model, first.reason) == (second.route, second.model, second.reason)


def test_resolve_model_rejects_an_unknown_route():
    with pytest.raises(RouterConfigError):
        resolve_model("cheap-plus", load_router_config())


def test_router_config_cannot_steer_retrieval():
    """Routing is a generate-side choice: no retrieve/mode keys belong here."""
    fields = set(load_router_config().model_dump())
    assert not {"retrieve", "retrieve_mode", "mode", "k", "bm25_top_n"} & fields


# ── config validation (fail loud, never a silently dead rule) ─────────────────


def test_unknown_when_key_is_a_config_error(tmp_path):
    path = write_config(tmp_path, rules=[{"when": {"always": True}, "route": "strong"}])
    with pytest.raises(RouterConfigError) as exc:
        load_router_config(path)
    assert "unknown" in str(exc.value)


def test_rule_route_must_exist_in_models(tmp_path):
    path = write_config(tmp_path, rules=[{"when": {"has_identifier": True}, "route": "medium"}])
    with pytest.raises(RouterConfigError) as exc:
        load_router_config(path)
    assert "medium" in str(exc.value)


def test_empty_when_is_rejected(tmp_path):
    path = write_config(tmp_path, rules=[{"when": {}, "route": "cheap"}])
    with pytest.raises(RouterConfigError):
        load_router_config(path)


def test_missing_config_file_is_a_config_error(tmp_path):
    with pytest.raises(RouterConfigError):
        load_router_config(tmp_path / "nope.yaml")


def test_llm_policy_is_refused_until_implemented(tmp_path):
    path = write_config(tmp_path, policy="llm")
    config = load_router_config(path)
    with pytest.raises(RouterConfigError) as exc:
        decide(extract_signals("What does E-4471 mean?", config=config), config=config)
    assert "not implemented" in str(exc.value)


# ── tracing ───────────────────────────────────────────────────────────────────


def test_router_decide_span_carries_the_contract(exporter):
    with start_trace("assemble", query="What does E-4471 mean?") as trace:
        decision = route_request(
            "What does E-4471 mean?", {"n_retrieved": 5, "retrieval_ran": True}, dry_run=True
        )

    records = exporter.for_trace(trace.trace_id)
    decide_span = [record for record in records if record.name == "router.decide"]
    assert len(decide_span) == 1
    span = decide_span[0]

    for key in ("route", "model", "reason", "intent_hint", "has_identifier", "dry_run"):
        assert key in span.attributes, key
    assert span.attributes["route"] == decision.route
    assert span.attributes["model"] == decision.model
    assert span.attributes["intent_hint"] == "lexical"
    assert span.attributes["has_identifier"] is True
    assert span.attributes["dry_run"] is True
    assert span.attributes["n_retrieved"] == 5
    assert "missing_context" not in span.attributes
    assert span.parent_span_id == trace.root.span_id
    assert span.status == "ok"
    assert span.duration_ms is not None


# ── generate: fallback, dry-run, cost ─────────────────────────────────────────


def test_dry_run_makes_no_call_and_writes_no_generate_span(exporter):
    class Exploding:
        def __getattr__(self, item):
            raise AssertionError("dry-run must not touch a client")

    with start_trace("assemble"):
        result = generate(
            GenerateRequest(prompt="x", model="test-cheap"),
            dry_run=True,
            client=Exploding(),
            config=config_for(),
        )

    assert result.output is None
    assert result.dry_run is True
    assert result.fallback_used is False
    assert result.input_tokens is None
    assert not [r for r in exporter.records if r.name == "generate"]


def test_primary_failure_returns_the_fallback_model(exporter):
    client = stub_client(fail_models={"test-cheap"})

    with start_trace("assemble") as trace:
        result = generate(
            GenerateRequest(prompt="hello", model="test-cheap"),
            dry_run=False,
            client=client,
            config=config_for(),
        )

    assert result.model == "test-fallback"
    assert result.requested_model == "test-cheap"
    assert result.fallback_used is True
    assert result.output == "stub output"
    assert "test-cheap is down" in result.primary_error
    assert [call["model"] for call in client.calls] == ["test-cheap", "test-fallback"]

    span = [r for r in exporter.for_trace(trace.trace_id) if r.name == "generate"][0]
    assert span.status == "ok", "a recovered fallback is not an error span"
    assert span.attributes["fallback_used"] is True
    assert span.attributes["model"] == "test-fallback"
    assert span.attributes["attempts"] == 2
    assert "test-cheap is down" in span.attributes["primary_error"]
    assert span.attributes["input_tokens"] == 12
    assert span.attributes["output_tokens"] == 5
    assert "missing_context" not in span.attributes
    assert "prompt" not in span.attributes  # include_prompt stays false


def test_happy_path_uses_the_routed_model_without_fallback(exporter):
    client = stub_client()
    with start_trace("assemble") as trace:
        result = generate(
            GenerateRequest(prompt="hello", model="test-strong"),
            dry_run=False,
            client=client,
            config=config_for(),
        )

    assert result.model == "test-strong" and result.fallback_used is False
    assert [call["model"] for call in client.calls] == ["test-strong"]
    span = [r for r in exporter.for_trace(trace.trace_id) if r.name == "generate"][0]
    assert span.attributes["fallback_used"] is False
    assert span.attributes["attempts"] == 1


def test_fallback_disabled_propagates_the_primary_error():
    with pytest.raises(RuntimeError):
        generate(
            GenerateRequest(prompt="hello", model="test-cheap"),
            dry_run=False,
            client=stub_client(fail_models={"test-cheap"}),
            config=config_for(fallback_on_error=False),
        )


def test_both_models_failing_raises_and_marks_the_span_error(exporter):
    with start_trace("assemble") as trace:
        with pytest.raises(RuntimeError):
            generate(
                GenerateRequest(prompt="hello", model="test-cheap"),
                dry_run=False,
                client=stub_client(fail_models={"test-cheap", "test-fallback"}),
                config=config_for(),
            )

    span = [r for r in exporter.for_trace(trace.trace_id) if r.name == "generate"][0]
    assert span.status == "error"
    assert "test-fallback is down" in span.error


def test_placeholder_fallback_means_no_fallback_not_fatal(exporter):
    """A real primary must still work when only the fallback is unconfigured."""
    config = config_for(models={
        "cheap": "test-cheap",
        "strong": "test-strong",
        "fallback": "PLACEHOLDER_FALLBACK_MODEL",
    })
    client = stub_client()

    with start_trace("assemble") as trace:
        result = generate(
            GenerateRequest(prompt="hello", model="test-cheap"),
            dry_run=False,
            client=client,
            config=config,
        )

    assert result.model == "test-cheap" and result.fallback_used is False
    assert [call["model"] for call in client.calls] == ["test-cheap"]
    span = [r for r in exporter.for_trace(trace.trace_id) if r.name == "generate"][0]
    assert span.attributes["fallback_configured"] is False
    assert span.attributes["attempts"] == 1


def test_placeholder_fallback_is_never_sent_to_the_api():
    """Primary failure with no usable fallback propagates — no bogus model call."""
    config = config_for(models={
        "cheap": "test-cheap",
        "strong": "test-strong",
        "fallback": "PLACEHOLDER_FALLBACK_MODEL",
    })
    client = stub_client(fail_models={"test-cheap"})

    with pytest.raises(RuntimeError):
        generate(
            GenerateRequest(prompt="hello", model="test-cheap"),
            dry_run=False,
            client=client,
            config=config,
        )

    assert [call["model"] for call in client.calls] == ["test-cheap"]


def test_placeholder_model_refuses_a_live_call():
    config = load_router_config()  # shipped config keeps placeholders
    with pytest.raises(RouterConfigError) as exc:
        generate(
            GenerateRequest(prompt="hello", model=config.models["strong"]),
            dry_run=False,
            client=stub_client(),
            config=config,
        )
    assert "placeholder" in str(exc.value).lower()


def test_usage_less_response_is_marked_missing_context_not_zeroed(exporter):
    with start_trace("assemble") as trace:
        result = generate(
            GenerateRequest(prompt="hello", model="test-cheap"),
            dry_run=False,
            client=stub_client(usage=False),
            config=config_for(),
        )

    assert result.input_tokens is None and result.output_tokens is None
    span = [r for r in exporter.for_trace(trace.trace_id) if r.name == "generate"][0]
    assert "input_tokens" not in span.attributes
    assert span.attributes["missing_context"] == ["input_tokens", "output_tokens"]


def test_cost_estimate_is_omitted_when_the_price_is_unknown(exporter):
    config = config_for(prices_usd_per_1m={"test-cheap": None})
    with start_trace("assemble") as trace:
        result = generate(
            GenerateRequest(prompt="hello", model="test-cheap"),
            dry_run=False,
            client=stub_client(),
            config=config,
        )

    assert result.cost_usd_estimate is None
    span = [r for r in exporter.for_trace(trace.trace_id) if r.name == "generate"][0]
    assert "cost_usd_estimate" not in span.attributes, "unknown price must be omitted, not 0.0"


def test_cost_estimate_handles_flat_and_split_prices():
    flat = config_for(prices_usd_per_1m={"test-cheap": 2.0})
    assert estimate_cost_usd("test-cheap", 1000, 1000, flat) == pytest.approx(2000 * 2.0 / 1_000_000)

    split = config_for(prices_usd_per_1m={"test-cheap": {"input": 1.0, "output": 3.0}})
    assert estimate_cost_usd("test-cheap", 1000, 500, split) == pytest.approx((1000 * 1.0 + 500 * 3.0) / 1_000_000)

    assert estimate_cost_usd("test-cheap", None, 500, split) is None
    assert estimate_cost_usd("not-priced", 1000, 1000, flat) is None


def test_cost_estimate_lands_on_the_generate_span_when_priced(exporter):
    config = config_for(prices_usd_per_1m={"test-cheap": {"input": 1.0, "output": 3.0}})
    with start_trace("assemble") as trace:
        result = generate(
            GenerateRequest(prompt="hello", model="test-cheap"),
            dry_run=False,
            client=stub_client(),
            config=config,
        )

    expected = (12 * 1.0 + 5 * 3.0) / 1_000_000
    assert result.cost_usd_estimate == pytest.approx(expected)
    span = [r for r in exporter.for_trace(trace.trace_id) if r.name == "generate"][0]
    assert span.attributes["cost_usd_estimate"] == pytest.approx(expected)


# ── golden set + suite + gate ─────────────────────────────────────────────────


def test_golden_set_matches_the_required_distribution():
    from contextlab.evals import suites_router

    cases = suites_router.load_golden()
    assert len(cases) >= 15
    assert all(case.expected_route in ("cheap", "strong", "fallback") for case in cases)

    by_intent = Counter(case.intent for case in cases)
    assert by_intent["version"] >= 3
    assert by_intent["table"] >= 2
    assert by_intent["paraphrase"] >= 2
    assert by_intent["negative"] >= 2

    identifiers = [case for case in cases if has_identifier(case.query)]
    assert len(identifiers) >= 4
    assert all(case.expected_route == "cheap" for case in identifiers)

    conflicts = [case for case in cases if case.memory_items]
    assert len(conflicts) >= 2
    assert all(case.expected_route == "strong" for case in conflicts)
    assert all(case.retrieve for case in conflicts), "a conflict needs real retrieval to exist"


def test_router_suite_meets_the_accuracy_gate():
    from contextlab.evals import suites_router

    results, metrics = suites_router.run_suite()

    assert metrics.n == len(results) >= 15
    assert metrics.metrics["route_accuracy"] >= 0.85
    assert metrics.metrics["intent_accuracy"] == 1.0, "golden intent labels must match the shipped policy"
    assert metrics.metrics["routes_strong"] > 0 and metrics.metrics["routes_cheap"] > 0
    assert all(result.trace_id for result in results)
    assert all(result.route in ("cheap", "strong", "fallback") for result in results)
    assert all(result.model for result in results)


def test_eval_case_trace_nests_the_router_decision(exporter, tmp_path):
    from contextlab.evals import suites_router

    golden = tmp_path / "router_one.jsonl"
    golden.write_text(json.dumps({
        "id": "rte900",
        "query": "What does E-4471 mean?",
        "intent": "lexical",
        "expected_route": "cheap",
        "retrieve": False,
    }) + "\n")

    results, metrics = suites_router.run_suite(golden_path=golden)
    assert metrics.metrics["route_accuracy"] == 1.0
    case_result = results[0]
    assert case_result.route == "cheap" and case_result.model

    records = exporter.for_trace(case_result.trace_id)
    decide_span = [r for r in records if r.name == "router.decide"][0]
    case_span = [r for r in records if r.name == "eval.case"][0]

    assert decide_span.parent_span_id == case_span.span_id
    assert "missing_context" not in decide_span.attributes
    assert case_span.attributes["route"] == "cheap"
    assert case_span.attributes["passed"] is True
    assert case_span.attributes["expected_route"] == "cheap"


def test_gate_reads_route_accuracy_not_case_passes():
    from contextlab.evals.runner import apply_gates, load_gates
    from contextlab.evals.types import CaseResult, SuiteMetrics

    gates = load_gates()
    assert gates["router"]["min_accuracy"] == 0.85

    cases = [CaseResult(id="rte001", suite="router", passed=True)]
    below = SuiteMetrics(n=10, n_pass=8, n_fail=2, metrics={"route_accuracy": 0.80})
    failing = apply_gates({"router": (cases, below)}, {"router": {"min_accuracy": 0.85}})
    assert failing[0].name == "gate_router_accuracy"
    assert failing[0].passed is False

    above = SuiteMetrics(n=10, n_pass=9, n_fail=1, metrics={"route_accuracy": 0.90})
    passing = apply_gates({"router": (cases, above)}, {"router": {"min_accuracy": 0.85}})
    assert passing[0].passed is True


# ── CLI ───────────────────────────────────────────────────────────────────────


def test_route_cli_dry_run_prints_route_reason_and_model(capsys):
    from contextlab.route import main

    assert main(["--query", "what does E-4471 mean", "--dry-run"]) == 0
    out = capsys.readouterr().out

    assert re.search(r"^route\s+cheap$", out, re.M)
    assert re.search(r"^model\s+gpt-4o-mini$", out, re.M)
    assert re.search(r"^reason\s+rule\[2\] has_identifier=true -> cheap$", out, re.M)
    assert re.search(r"^dry_run\s+true$", out, re.M)
    assert "intent_hint=lexical" in out
    assert "no model was called" in out


def test_route_cli_version_query_routes_strong(capsys):
    from contextlab.route import main

    assert main(["--query", "What was the refund window before we changed it?", "--generate"]) == 0
    captured = capsys.readouterr()

    assert re.search(r"^route\s+strong$", captured.out, re.M)
    assert "intent_hint=version" in captured.out
    # --generate on a placeholder model must skip loudly, never crash or call HTTP
    assert "generate skipped" in captured.err
    assert "placeholder" in captured.err.lower()

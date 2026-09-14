"""Slice 6 cache tests: fingerprint, exact/semantic layers, guards, false hits.

No test needs an LLM or a network. Embeddings for the focused tests come from a stub
vector space (so distances are controlled), and the answer-path tests use a stub
generate client.
"""
import json
from collections import Counter

import pytest

from contextlab.cache import (
    Cache,
    CacheConfig,
    CacheFingerprint,
    CacheStore,
    cosine,
    corpus_version,
    fingerprint_for,
    fingerprint_from_dict,
    guard_reason,
    hash_text,
    load_cache_config,
    new_entry,
    normalize_query,
)
from contextlab.cache.answer import answer_with_cache
from contextlab.cache.lookup import capitalized_entities, identifiers_in
from contextlab.trace import InMemoryExporter, configure, reset, start_trace
from contextlab.types import AssembledContext


def cache_config(**overrides) -> CacheConfig:
    """Shipped config (threshold, guards) with per-test path/threshold overrides."""
    data = load_cache_config().model_dump()
    data.update(overrides)
    return CacheConfig(**data)


def fingerprint(budget_tokens: int = 800, retrieve_mode: str = "hybrid", **extra) -> CacheFingerprint:
    """Deterministic fingerprint — no dependency on the repo's corpus files."""
    return fingerprint_for(
        retrieve_mode=retrieve_mode,
        budget_tokens=budget_tokens,
        system="you are a test system prompt",
        corpus=extra.pop("corpus", "corpus-test"),
        **extra,
    )


class StubEncoder:
    """Hand-written vector space: exact control over every distance."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors
        self.calls: list[str] = []

    def encode(self, text: str) -> list[float]:
        self.calls.append(text)
        if text not in self.vectors:
            raise AssertionError(f"unexpected encode({text!r})")
        return self.vectors[text]


def stub_client(output: str = "generated answer", fail: bool = False):
    """A client shaped like the OpenAI SDK."""
    class _Completions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if fail:
                raise RuntimeError("api is down")
            return type("Response", (), {
                "usage": type("Usage", (), {"prompt_tokens": 11, "completion_tokens": 4})(),
                "choices": [type("Choice", (), {"message": type("Message", (), {"content": output})()})()],
            })()

    completions = _Completions()
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    client.calls = completions.calls
    return client


def fake_ctx(query: str = "What does E-4471 mean?", citations=("error_codes::c0001",)) -> AssembledContext:
    """An assembled context without paying for retrieval."""
    return AssembledContext(
        query=query,
        budget_tokens=800,
        prompt_tokens=42,
        prompt="# Instructions\nbe helpful\n\n# User\n" + query,
        blocks=[],
        dropped=[],
        citations=list(citations),
        retrieval={},
        settings={},
    )


@pytest.fixture
def exporter():
    """In-memory exporter — span assertions never touch data/traces/."""
    memory = InMemoryExporter()
    configure(exporter=memory)
    yield memory
    reset()


# ── fingerprint + normalization (feat-c1) ─────────────────────────────────────


def test_normalize_query_casefolds_whitespace_and_punctuation():
    assert normalize_query("  What   does E-4471 mean?  ") == "what does e-4471 mean"
    assert normalize_query("What does E-4471 mean") == normalize_query("what does e-4471 mean?!")


def test_corpus_version_tracks_the_chunks_file(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text('{"chunk_id": "a"}\n')
    first = corpus_version(chunks)
    assert len(first) == 16

    chunks.write_text('{"chunk_id": "a"}\n{"chunk_id": "b"}\n')
    assert corpus_version(chunks) != first, "a re-ingest must invalidate cached entries"

    assert corpus_version(tmp_path / "missing.jsonl") == "no-corpus"


def test_fingerprint_key_includes_route_and_model_only_when_known():
    bare = fingerprint()
    assert "route=" not in bare.key() and "model=" not in bare.key()

    routed = fingerprint(route="cheap", model="gpt-4o-mini")
    assert "route=cheap" in routed.key() and "model=gpt-4o-mini" in routed.key()
    assert routed.key() != bare.key()


def test_fingerprint_differs_from_names_the_changed_fields():
    base = fingerprint()
    assert base.differs_from(fingerprint()) == []
    assert base.differs_from(fingerprint(budget_tokens=400)) == ["budget_tokens"]
    assert base.differs_from(fingerprint(retrieve_mode="bm25")) == ["retrieve_mode"]
    assert set(base.differs_from(fingerprint(route="cheap", model="m"))) == {"route", "model"}


def test_fingerprint_from_dict_rejects_unknown_fields():
    assert fingerprint_from_dict({"retrieve_mode": "bm25", "budget_tokens": 400}).budget_tokens == 400
    with pytest.raises(ValueError) as exc:
        fingerprint_from_dict({"budget": 400})
    assert "unknown fingerprint field" in str(exc.value)


def test_same_query_different_budget_is_a_miss(tmp_path):
    """feat-c1: a fingerprint mismatch is always a miss, even for identical text."""
    config = cache_config(path=str(tmp_path / "cache.jsonl"))
    store = CacheStore(config.resolved_path())
    cache = Cache(config, store, encoder=StubEncoder({"What does E-4471 mean?": [1.0, 0.0]}))

    cache.write("What does E-4471 mean?", fingerprint(budget_tokens=800), answer="30 days.")

    lookup = cache.lookup("What does E-4471 mean?", fingerprint(budget_tokens=400))
    assert lookup.hit is False
    assert lookup.reason == "fingerprint_mismatch"
    assert "budget_tokens" in (lookup.detail or "")
    assert lookup.entry is None


def test_identical_text_and_fingerprint_hits_via_the_exact_layer(tmp_path):
    config = cache_config(path=str(tmp_path / "cache.jsonl"))
    cache = Cache(
        config,
        CacheStore(config.resolved_path()),
        encoder=StubEncoder({"What does E-4471 mean?": [1.0, 0.0]}),
    )

    entry = cache.write("What does E-4471 mean?", fingerprint(), answer="Inventory reservation expired.")
    lookup = cache.lookup("  what does E-4471 mean?!", fingerprint())

    assert lookup.hit is True
    assert lookup.reason == "exact"
    assert lookup.score is None, "the exact layer computes no similarity"
    assert lookup.entry is not None and lookup.entry.cache_id == entry.cache_id
    assert lookup.entry.answer == "Inventory reservation expired."


def test_exact_text_with_a_different_mode_is_fingerprint_mismatch(tmp_path):
    config = cache_config(path=str(tmp_path / "cache.jsonl"))
    cache = Cache(
        config,
        CacheStore(config.resolved_path()),
        encoder=StubEncoder({"Ground shipping SLA target?": [1.0, 0.0]}),
    )

    cache.write("Ground shipping SLA target?", fingerprint(retrieve_mode="hybrid"), answer="5 days.")
    lookup = cache.lookup("Ground shipping SLA target?", fingerprint(retrieve_mode="bm25"))

    assert lookup.hit is False and lookup.reason == "fingerprint_mismatch"


def test_require_fingerprint_false_lets_a_budget_change_hit(tmp_path):
    """The mutation lever: switching the fingerprint off is what creates false hits."""
    config = cache_config(path=str(tmp_path / "cache.jsonl"), require_fingerprint=False)
    cache = Cache(
        config,
        CacheStore(config.resolved_path()),
        encoder=StubEncoder({"Professional plan user limit?": [1.0, 0.0]}),
    )

    cache.write("Professional plan user limit?", fingerprint(budget_tokens=800), answer="25 users.")
    lookup = cache.lookup("Professional plan user limit?", fingerprint(budget_tokens=400))

    assert lookup.hit is True and lookup.reason == "exact"


# ── guards ────────────────────────────────────────────────────────────────────


def test_identifier_guard_blocks_one_digit_apart():
    reason = guard_reason("What does E-4472 mean?", "What does E-4471 mean?", load_cache_config())
    assert reason is not None and reason.startswith("guard:identifier_mismatch")


def test_entity_guard_blocks_a_different_capitalized_entity():
    config = load_cache_config()
    assert guard_reason(
        "What is the SLA for ground shipping to the East Coast?",
        "What is the SLA for ground shipping to the West Coast?",
        config,
    ) == "guard:entity_mismatch east"
    assert guard_reason(
        "How much does the Professional plan cost?", "How much does the Starter plan cost?", config
    ) == "guard:entity_mismatch professional"


def test_version_conflict_guard_blocks_old_vs_current():
    config = load_cache_config()
    assert guard_reason(
        "What was the old refund window before the policy change?",
        "What is the current refund window?",
        config,
    ) == "guard:version_conflict"
    # same group: "current" vs "now" is the same policy version
    assert guard_reason("How long is the refund window right now?", "What is the current refund window?", config) is None


def test_number_guard_requires_probe_numbers_in_the_seed():
    config = load_cache_config()
    assert guard_reason("Is the refund 60 days?", "Is the refund 30 days?", config) is not None
    assert guard_reason("Is the refund 30 days?", "Is the refund 30 days?", config) is None


def test_guards_only_fire_on_tokens_the_probe_introduces():
    """A guard can cost a hit, never cause a wrong one."""
    config = load_cache_config()
    # seed is the more specific question; the probe drops the entity
    assert guard_reason("What does E-4471 mean?", "What does E-4471 mean exactly?", config) is None
    assert guard_reason("Summarize it", "What does E-4471 mean?", config) is None


def test_capitalized_entities_skips_the_first_word_and_acronyms():
    assert capitalized_entities("Professional plan user limit?") == set(), "sentence-case first word"
    assert capitalized_entities("What is the SLA for West Coast shipping?") == {"west", "coast"}
    assert capitalized_entities("what is the sla for west coast shipping") == set(), "lowercase = no entity"


# ── semantic layer (feat-c2) ──────────────────────────────────────────────────


def test_semantic_hit_above_threshold_picks_the_nearest_entry(tmp_path):
    config = cache_config(path=str(tmp_path / "cache.jsonl"), threshold=0.9)
    encoder = StubEncoder({
        "What does E-4471 mean?": [1.0, 0.0],
        "Meaning of error code E-4471": [0.95, 0.05],
        "Unrelated question": [0.0, 1.0],
    })
    cache = Cache(config, CacheStore(config.resolved_path()), encoder=encoder)

    cache.write("What does E-4471 mean?", fingerprint(), answer="Inventory reservation expired.")
    cache.write("Unrelated question", fingerprint(), answer="Something else.")

    lookup = cache.lookup("Meaning of error code E-4471", fingerprint())
    assert lookup.hit is True
    assert lookup.reason == "semantic"
    assert lookup.score == pytest.approx(cosine([0.95, 0.05], [1.0, 0.0]), rel=1e-6)
    assert lookup.entry is not None and "E-4471" in lookup.entry.query


def test_semantic_miss_below_threshold_records_the_rejected_score(tmp_path):
    config = cache_config(path=str(tmp_path / "cache.jsonl"), threshold=0.95)
    encoder = StubEncoder({"seed question": [1.0, 0.0], "probe question": [0.5, 0.5]})
    cache = Cache(config, CacheStore(config.resolved_path()), encoder=encoder)

    cache.write("seed question", fingerprint(), answer="answer")
    lookup = cache.lookup("probe question", fingerprint())

    assert lookup.hit is False
    assert lookup.reason == "miss"
    assert lookup.score == pytest.approx(0.7071067, rel=1e-5)
    assert "0.95" in (lookup.detail or "")


def test_similarity_is_cosine_not_dot_product(tmp_path):
    config = cache_config(path=str(tmp_path / "cache.jsonl"), threshold=0.99)
    encoder = StubEncoder({"a": [3.0, 4.0], "b": [6.0, 8.0]})
    cache = Cache(config, CacheStore(config.resolved_path()), encoder=encoder)

    cache.write("a", fingerprint(), answer="scaled copy")
    lookup = cache.lookup("b", fingerprint())

    assert lookup.score == pytest.approx(1.0), "unnormalized vectors must still score 1.0"
    assert lookup.hit is True


def test_entries_without_stored_vectors_are_re_embedded(tmp_path):
    config = cache_config(path=str(tmp_path / "cache.jsonl"), threshold=0.9)
    encoder = StubEncoder({"seed question": [1.0, 0.0], "probe question": [1.0, 0.0]})
    store = CacheStore(config.resolved_path())
    cache = Cache(config, store, encoder=encoder)

    # written without a vector, the way a sidecar-less JSONL row may look
    store.append(new_entry(query="seed question", fingerprint=fingerprint(), answer="answer"))
    assert store.entries()[0].embedding is None

    lookup = cache.lookup("probe question", fingerprint())
    assert lookup.hit is True and lookup.reason == "semantic"


def test_empty_store_misses_without_embedding_anything(tmp_path):
    config = cache_config(path=str(tmp_path / "never-written.jsonl"))
    encoder = StubEncoder({})
    cache = Cache(config, CacheStore(config.resolved_path()), encoder=encoder)

    lookup = cache.lookup("anything", fingerprint())
    assert lookup.hit is False and lookup.reason == "miss"
    assert lookup.detail == "cache is empty"
    assert encoder.calls == [], "an empty cache must not load the model"


# ── store ─────────────────────────────────────────────────────────────────────


def test_store_roundtrip_is_one_object_per_line(tmp_path):
    path = tmp_path / "cache.jsonl"
    store = CacheStore(path)
    entry = new_entry(
        query="What does E-4471 mean?",
        fingerprint=fingerprint(),
        answer="Inventory reservation expired.",
        citations=["error_codes::c0001"],
        embedding=[0.1, 0.2, 0.3],
    )
    store.append(entry)

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["cache_id"] == entry.cache_id
    assert row["fingerprint"]["budget_tokens"] == 800
    assert row["citations"] == ["error_codes::c0001"]

    reread = CacheStore(path).entries()
    assert len(reread) == 1 and reread[0].cache_id == entry.cache_id
    assert reread[0].embedding == [0.1, 0.2, 0.3]


def test_cache_id_is_deterministic_per_request():
    def entry_for(budget):
        return new_entry(
            query="What does E-4471 mean?",
            fingerprint=fingerprint(budget_tokens=budget),
            answer="a",
        )

    assert entry_for(800).cache_id == entry_for(800).cache_id
    assert entry_for(800).cache_id != entry_for(400).cache_id


def test_store_skips_unparseable_lines(tmp_path):
    path = tmp_path / "cache.jsonl"
    store = CacheStore(path)
    store.append(new_entry(query="q", fingerprint=fingerprint(), answer="a"))

    with open(path, "a") as f:
        f.write('{"broken": \n')

    assert len(CacheStore(path).entries()) == 1


def test_store_stats_report_corpus_versions(tmp_path):
    path = tmp_path / "cache.jsonl"
    store = CacheStore(path)
    store.append(new_entry(query="a", fingerprint=fingerprint(), answer="x"))
    store.append(new_entry(query="b", fingerprint=fingerprint(corpus="corpus-old"), answer="y"))

    stats = store.stats()
    assert stats["entries"] == 2
    assert stats["by_corpus_version"] == {"corpus-test": 1, "corpus-old": 1}
    assert stats["by_retrieve_mode"] == {"hybrid": 2}


def test_clear_removes_everything(tmp_path):
    path = tmp_path / "cache.jsonl"
    store = CacheStore(path)
    store.append(new_entry(query="a", fingerprint=fingerprint(), answer="x"))
    store.clear()
    assert store.entries() == []
    assert not path.exists()


# ── tracing (feat-c4) ─────────────────────────────────────────────────────────


def test_cache_lookup_span_carries_hit_score_and_reason(exporter, tmp_path):
    config = cache_config(path=str(tmp_path / "cache.jsonl"), threshold=0.9)
    encoder = StubEncoder({"What does E-4471 mean?": [1.0, 0.0]})
    cache = Cache(config, CacheStore(config.resolved_path()), encoder=encoder)

    with start_trace("assemble") as trace:
        cache.write("What does E-4471 mean?", fingerprint(), answer="Inventory reservation expired.")
        cache.lookup("What does E-4471 mean?", fingerprint())

    records = exporter.for_trace(trace.trace_id)
    lookup_span = [r for r in records if r.name == "cache.lookup"][0]
    write_span = [r for r in records if r.name == "cache.write"][0]

    assert lookup_span.attributes["hit"] is True
    assert lookup_span.attributes["score"] is None  # exact layer
    assert lookup_span.attributes["reason"] == "exact"
    assert lookup_span.attributes["cache_id"] == write_span.attributes["cache_id"]
    assert "missing_context" not in lookup_span.attributes
    assert "missing_context" not in write_span.attributes
    assert lookup_span.parent_span_id == trace.root.span_id
    assert write_span.parent_span_id == trace.root.span_id
    assert write_span.attributes["citations"] == 0


# ── answer path wiring (feat-c4) ──────────────────────────────────────────────


def test_miss_generates_and_writes_then_the_next_call_hits(tmp_path):
    """The point of the slice: second identical request never reaches the model."""
    config = cache_config(path=str(tmp_path / "cache.jsonl"))
    store = CacheStore(config.resolved_path())
    cache = Cache(config, store, encoder=StubEncoder({"What does E-4471 mean?": [1.0, 0.0]}))
    client = stub_client(output="Inventory reservation expired.")

    with start_trace("assemble"):
        first = answer_with_cache(
            "What does E-4471 mean?",
            system="you are a test system prompt",
            ctx=fake_ctx(),
            cache=cache,
            dry_run=False,
            generate_client=client,
        )
    assert first.lookup.hit is False
    assert first.answer == "Inventory reservation expired."
    assert first.citations == ["error_codes::c0001"]
    assert first.wrote_cache_id
    assert len(client.calls) == 1

    with start_trace("assemble"):
        second = answer_with_cache(
            "What does E-4471 mean?",
            system="you are a test system prompt",
            ctx=fake_ctx(),
            cache=cache,
            dry_run=False,
            generate_client=client,
        )
    assert second.lookup.hit is True
    assert second.answer == "Inventory reservation expired."
    assert second.citations == ["error_codes::c0001"]
    assert second.skipped_generate is True
    assert len(client.calls) == 1, "a hit must not call the model again"


def test_hit_never_touches_the_client(tmp_path):
    class Exploding:
        def __getattr__(self, item):
            raise AssertionError("a cache hit must not build a client")

    from contextlab.router import load_router_config

    config = cache_config(path=str(tmp_path / "cache.jsonl"))
    cache = Cache(config, CacheStore(config.resolved_path()), encoder=StubEncoder({"q": [1.0, 0.0]}))

    # the fingerprint the answer path will compute for "q": default route (cheap) +
    # its model, and the live corpus version the answer path will hash
    cache.write(
        "q",
        fingerprint_for(
            retrieve_mode="hybrid",
            budget_tokens=800,
            system="you are a test system prompt",
            corpus=corpus_version(),
            route="cheap",
            model=load_router_config().models["cheap"],
        ),
        answer="cached",
    )

    with start_trace("assemble"):
        result = answer_with_cache(
            "q",
            system="you are a test system prompt",
            ctx=fake_ctx("q"),
            cache=cache,
            dry_run=False,
            generate_client=Exploding(),
        )
    assert result.lookup.hit is True
    assert result.answer == "cached"


def test_dry_run_writes_nothing_and_calls_no_model(tmp_path):
    config = cache_config(path=str(tmp_path / "cache.jsonl"))
    store = CacheStore(config.resolved_path())
    cache = Cache(config, store, encoder=StubEncoder({"q": [1.0, 0.0]}))

    with start_trace("assemble"):
        result = answer_with_cache(
            "q",
            system="you are a test system prompt",
            ctx=fake_ctx("q"),
            cache=cache,
            dry_run=True,
        )
    assert result.lookup.hit is False
    assert result.answer is None
    assert result.wrote_cache_id is None
    assert store.entries() == []


def test_generation_failure_is_captured_and_writes_nothing(tmp_path):
    config = cache_config(path=str(tmp_path / "cache.jsonl"))
    store = CacheStore(config.resolved_path())
    cache = Cache(config, store, encoder=StubEncoder({"q": [1.0, 0.0]}))

    with start_trace("assemble"):
        result = answer_with_cache(
            "q",
            system="you are a test system prompt",
            ctx=fake_ctx("q"),
            cache=cache,
            dry_run=False,
            generate_client=stub_client(fail=True),
        )

    assert result.generate_error is not None and "api is down" in result.generate_error
    assert result.answer is None
    assert result.wrote_cache_id is None
    assert store.entries() == [], "no answer means nothing to cache"


def test_disabled_cache_records_no_lookup_span(exporter, tmp_path):
    config = cache_config(enabled=False, path=str(tmp_path / "cache.jsonl"))
    with start_trace("assemble") as trace:
        result = answer_with_cache(
            "q",
            system="you are a test system prompt",
            ctx=fake_ctx("q"),
            cache_config=config,
            dry_run=True,
        )

    assert result.lookup.hit is False
    assert result.lookup.detail == "cache disabled"
    assert [r for r in exporter.for_trace(trace.trace_id) if r.name == "cache.lookup"] == []


# ── golden set, suite, gates ──────────────────────────────────────────────────


def test_golden_set_matches_the_briefs_distribution():
    from contextlab.evals import suites_cache

    cases = suites_cache.load_golden()
    assert len(cases) >= 12
    types = Counter(case.case_type for case in cases)
    assert types["should_hit"] >= 7 and types["must_miss"] >= 5

    should = [c for c in cases if c.case_type == "should_hit"]
    exact = [c for c in should if normalize_query(c.seed_query) == normalize_query(c.probe_query)]
    assert len(exact) == 2, "two exact-repeat rows exercise the normalized-query layer"

    paraphrases = [c for c in should if normalize_query(c.seed_query) != normalize_query(c.probe_query)]
    assert len(paraphrases) >= 5

    notes = " ".join((c.notes or "").lower() for c in cases)
    for keyword in ("current", "old", "e-4472", "west", "retrieve_mode", "budget"):
        assert keyword in notes or keyword in " ".join(
            (c.seed_query + c.probe_query).lower() for c in cases
        ), keyword


def test_cache_suite_meets_both_gates():
    from contextlab.evals import suites_cache

    results, metrics = suites_cache.run_suite()

    assert metrics.n == len(results) >= 12
    assert metrics.metrics["should_hit_rate"] >= 0.80
    assert metrics.metrics["false_hit_rate"] == 0.0
    assert metrics.metrics["n_should_hit"] >= 7 and metrics.metrics["n_must_miss"] >= 5
    assert metrics.metrics["min_hit_score"] is not None
    assert metrics.metrics["max_miss_score"] is not None
    # the bands are inverted: the closest must-miss scores above the lowest hit
    assert metrics.metrics["max_miss_score"] > metrics.metrics["min_hit_score"]
    assert metrics.metrics["reasons"]["exact"] == 2
    assert any(reason.startswith("guard:") for reason in metrics.metrics["reasons"])
    assert all(result.trace_id for result in results)
    assert all(result.checks for result in results)


def test_cache_suite_leaves_the_real_store_alone():
    from contextlab.cache.config import PROJECT_ROOT
    from contextlab.evals import suites_cache

    real = PROJECT_ROOT / "data" / "cache.jsonl"  # the repo store, not the test one
    before = (real.exists(), real.stat().st_mtime if real.exists() else None)

    suites_cache.run_suite()

    after = (real.exists(), real.stat().st_mtime if real.exists() else None)
    assert before == after, "the suite must seed a temp store, not data/cache.jsonl"


def test_env_override_redirects_the_store(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEXTLAB_CACHE_PATH", str(tmp_path / "other.jsonl"))
    assert load_cache_config().resolved_path() == tmp_path / "other.jsonl"


def test_env_override_can_disable_the_cache(monkeypatch):
    monkeypatch.setenv("CONTEXTLAB_CACHE_ENABLED", "0")
    assert load_cache_config().enabled is False
    monkeypatch.setenv("CONTEXTLAB_CACHE_ENABLED", "1")
    assert load_cache_config().enabled is True


def test_answer_path_writes_to_the_env_configured_store(tmp_path, monkeypatch):
    """Regression: only the configured store is used, never a stray real one."""
    monkeypatch.setenv("CONTEXTLAB_CACHE_PATH", str(tmp_path / "env_cache.jsonl"))
    config = load_cache_config()
    assert config.resolved_path() == tmp_path / "env_cache.jsonl"

    cache = Cache(config, CacheStore(config.resolved_path()), encoder=StubEncoder({"q": [1.0, 0.0]}))
    with start_trace("assemble"):
        first = answer_with_cache(
            "q",
            system="you are a test system prompt",
            ctx=fake_ctx("q"),
            cache=cache,
            dry_run=False,
            generate_client=stub_client(output="env answer"),
        )
        second = answer_with_cache(
            "q",
            system="you are a test system prompt",
            ctx=fake_ctx("q"),
            cache=cache,
            dry_run=False,
            generate_client=stub_client(output="should not run"),
        )

    assert first.wrote_cache_id
    assert second.lookup.hit is True and second.answer == "env answer"
    assert (tmp_path / "env_cache.jsonl").exists()


def test_eval_case_trace_nests_cache_spans(exporter):
    from contextlab.evals import suites_cache

    results, _ = suites_cache.run_suite()
    records = exporter.for_trace(results[0].trace_id)
    lookup_spans = [r for r in records if r.name == "cache.lookup"]
    case_spans = [r for r in records if r.name == "eval.case"]

    assert len(case_spans) == 1 and lookup_spans
    assert lookup_spans[0].parent_span_id == case_spans[0].span_id
    for span in lookup_spans:
        assert "missing_context" not in span.attributes, span.attributes
        assert "hit" in span.attributes and "reason" in span.attributes


def test_mutation_require_fingerprint_false_creates_false_hits():
    """Recorded mutation (progress.md): the fingerprint is load-bearing."""
    from contextlab.evals import suites_cache

    shipped = suites_cache.run_suite()[1].metrics
    broken = suites_cache.run_suite(config=cache_config(require_fingerprint=False))[1].metrics

    assert shipped["false_hit_rate"] == 0.0
    assert broken["false_hit_rate"] > 0.0, "breaking the fingerprint must break the cache"
    assert broken["should_hit_rate"] >= shipped["should_hit_rate"]


def test_mutation_high_threshold_kills_the_hit_rate():
    """Recorded mutation (progress.md): raising the threshold does not buy safety."""
    from contextlab.evals import suites_cache

    broken = suites_cache.run_suite(config=cache_config(threshold=0.99))[1].metrics
    assert broken["should_hit_rate"] < 0.80
    assert broken["false_hit_rate"] == 0.0


def test_gates_read_should_hit_and_false_hit_rates():
    from contextlab.evals.runner import apply_gates, load_gates
    from contextlab.evals.types import CaseResult, SuiteMetrics

    gates = load_gates()
    assert gates["cache"]["min_should_hit_rate"] == 0.80
    assert gates["cache"]["max_false_hit_rate"] == 0.00

    cases = [CaseResult(id="c001", suite="cache", passed=True)]
    bad = SuiteMetrics(n=14, n_pass=13, n_fail=1, metrics={"should_hit_rate": 0.75, "false_hit_rate": 0.0})
    gates_out = apply_gates({"cache": (cases, bad)}, {"cache": {"min_should_hit_rate": 0.80, "max_false_hit_rate": 0.0}})
    assert {g.name: g.passed for g in gates_out} == {"gate_cache_should_hit": False, "gate_cache_false_hit": True}

    worse = SuiteMetrics(n=14, n_pass=11, n_fail=3, metrics={"should_hit_rate": 0.875, "false_hit_rate": 0.33})
    gates_out = apply_gates({"cache": (cases, worse)}, {"cache": {"min_should_hit_rate": 0.80, "max_false_hit_rate": 0.0}})
    assert {g.name: g.passed for g in gates_out} == {"gate_cache_should_hit": True, "gate_cache_false_hit": False}


def test_hash_text_is_short_and_stable():
    assert hash_text("abc") == hash_text("abc")
    assert len(hash_text("abc")) == 16
    assert hash_text("abc") != hash_text("abd")

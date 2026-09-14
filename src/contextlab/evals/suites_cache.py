"""Cache suite — hit rate on paraphrases, false-hit rate on near-duplicates.

Offline by construction: every entry is seeded from the golden file and no model is
asked for an answer. Embeddings come from the same local sentence-transformers model
dense retrieval uses, so the suite measures similarity, not a live LLM.

Two numbers decide whether the cache ships:
  should_hit_rate  = fraction of same-answer paraphrases that hit  (gate >= 0.80)
  false_hit_rate   = fraction of must-miss pairs that hit         (gate == 0.00)
The suite also reports min_hit_score and max_miss_score, because those bands are how
the threshold gets chosen — and they overlap (see progress.md).
"""
import json
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from contextlab.cache.config import CacheConfig, load_cache_config
from contextlab.cache.fingerprint import fingerprint_from_dict
from contextlab.cache.lookup import Cache
from contextlab.cache.store import CacheStore
from contextlab.evals.types import CaseResult, CheckResult, EvalCase, SuiteMetrics
from contextlab.trace import start_trace

CASE_TYPES = ("should_hit", "must_miss")


class CacheCase(EvalCase):
    """Golden row plus the fields only the cache suite needs.

    A subclass so Slice 3's EvalCase is untouched; `type` in the golden file is read
    as `case_type` here.
    """

    query: str = ""  # unused by this suite; the probe is what gets looked up
    case_type: str = ""
    seed_query: str = ""
    probe_query: str = ""
    seed_answer: str = ""
    fingerprint: dict = {}
    probe_fingerprint: dict = {}


def load_golden(path: str | Path = "evals/cache_golden.jsonl") -> list[CacheCase]:
    """Load cache golden rows, mapping the file's `type` to `case_type`."""
    cases: list[CacheCase] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            data["suite"] = "cache"
            data["case_type"] = data.pop("type", data.get("case_type", ""))
            data["query"] = data.get("probe_query", "")
            cases.append(CacheCase(**data))
    return cases


def run_suite(
    golden_path: str | Path = "evals/cache_golden.jsonl",
    config: Optional[CacheConfig] = None,
    workdir: Optional[str | Path] = None,
) -> tuple[list[CaseResult], SuiteMetrics]:
    """Seed each case, probe it, and grade the lookup. Never touches data/cache.jsonl."""
    config = config or load_cache_config()
    cases = load_golden(golden_path)
    results: list[CaseResult] = []

    n_should = n_should_hits = 0
    n_must = n_must_hits = 0
    hit_scores: list[float] = []
    miss_scores: list[float] = []
    reasons: Counter = Counter()

    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        store = CacheStore(Path(tmp) / "cache.jsonl")
        cache = Cache(config, store)

        for case in cases:
            if case.case_type not in CASE_TYPES:
                raise ValueError(
                    f"{case.id}: type must be one of {CASE_TYPES}, got {case.case_type!r}"
                )

            expect_hit = case.case_type == "should_hit"
            with start_trace(
                "eval.case", case_id=case.id, suite="cache", query=case.probe_query
            ) as trace:
                t0 = time.perf_counter()

                # one entry per case: cases must not see each other's seeds
                store.clear()
                seed_fp = fingerprint_from_dict(case.fingerprint)
                probe_fp = fingerprint_from_dict(case.probe_fingerprint or case.fingerprint)

                cache.write(
                    case.seed_query,
                    seed_fp,
                    answer=case.seed_answer,
                    citations=case.expected_citations or [],
                )
                lookup = cache.lookup(case.probe_query, probe_fp)

                correct = lookup.hit == expect_hit
                reasons[lookup.reason.split(" ")[0]] += 1

                if expect_hit:
                    n_should += 1
                    n_should_hits += int(lookup.hit)
                    if lookup.hit and lookup.score is not None:
                        hit_scores.append(lookup.score)
                else:
                    n_must += 1
                    n_must_hits += int(lookup.hit)
                    if lookup.score is not None:
                        miss_scores.append(lookup.score)

                score_text = f"{lookup.score:.4f}" if lookup.score is not None else "-"
                checks = [CheckResult(
                    name=case.case_type,
                    passed=correct,
                    detail=(
                        f"{'hit' if lookup.hit else 'miss'} reason={lookup.reason} "
                        f"score={score_text} :: {lookup.detail or ''}"
                    ),
                    score=lookup.score,
                )]

                trace.set_attributes(
                    passed=correct,
                    case_type=case.case_type,
                    expect_hit=expect_hit,
                    hit=lookup.hit,
                    reason=lookup.reason,
                    score=lookup.score,
                    cache_id=lookup.entry.cache_id if lookup.entry is not None else None,
                    threshold=config.threshold,
                )

                results.append(CaseResult(
                    id=case.id,
                    suite="cache",
                    passed=correct,
                    checks=checks,
                    cited_ids=list(case.expected_citations or []),
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    trace_id=trace.trace_id,
                ))

    metrics = {
        "should_hit_rate": n_should_hits / n_should if n_should else 0.0,
        "false_hit_rate": n_must_hits / n_must if n_must else 0.0,
        "n_should_hit": n_should,
        "n_must_miss": n_must,
        "min_hit_score": min(hit_scores) if hit_scores else None,
        "max_miss_score": max(miss_scores) if miss_scores else None,
        "threshold": config.threshold,
        "require_fingerprint": config.require_fingerprint,
        "reasons": dict(reasons),
    }

    sm = SuiteMetrics(
        n=len(cases),
        n_pass=sum(1 for r in results if r.passed),
        n_fail=sum(1 for r in results if not r.passed),
        metrics=metrics,
    )
    return results, sm

"""Retrieval suite — wraps existing eval_retrieval.

Each case runs inside its own root trace named "eval.case", so
`python -m contextlab.trace show --case <id>` prints that case's span tree
(including the retrieve.* hops and the chunk ids they returned).
"""
import json
import time
from pathlib import Path

from contextlab.evals.types import CaseResult, CheckResult, EvalCase, SuiteMetrics
from contextlab.retrieve import Retriever
from contextlab.config import get_config
from contextlab.trace import start_trace


def load_golden(path: str | Path = "evals/retrieval_golden.jsonl") -> list[EvalCase]:
    """Load retrieval golden cases."""
    cases = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            d["suite"] = "retrieval"  # override suite from file
            cases.append(EvalCase(**d))
    return cases


def run_suite(golden_path: str | Path = "evals/retrieval_golden.jsonl") -> tuple[list[CaseResult], SuiteMetrics]:
    """Run retrieval suite. Reuses existing eval_retrieval logic."""
    config = get_config()
    cases = load_golden(golden_path)
    results: list[CaseResult] = []

    retriever = Retriever(
        chunk_tokens=config["chunk_tokens"],
        chunk_overlap_tokens=config["chunk_overlap_tokens"],
        encoding=config["encoding"],
        embedding_model=config["embedding_model"],
        bm25_top_n=config["bm25_top_n"],
        dense_top_n=config["dense_top_n"],
        rrf_k=config["rrf_k"],
        hybrid_k=config["hybrid_k"],
    )

    # Load chunk ids for citation validity check
    chunk_ids: set[str] = set()
    chunk_path = Path("data/chunks.jsonl")
    if chunk_path.exists():
        with open(chunk_path) as f:
            for line in f:
                d = json.loads(line)
                chunk_ids.add(d["chunk_id"])

    mode_recalls = {"bm25": 0.0, "dense": 0.0, "hybrid": 0.0}
    intent_recalls: dict[str, dict[str, float]] = {}

    for case in cases:
        with start_trace(
            "eval.case", case_id=case.id, suite="retrieval", query=case.query
        ) as trace:
            t0 = time.perf_counter()
            checks: list[CheckResult] = []

            # Run hybrid (primary mode)
            request = type("Request", (), {"query": case.query, "k": 5, "mode": "hybrid", "filters": None})()
            response = retriever.retrieve(request)
            hit_ids = [h.chunk_id for h in response.hits]
            latency_ms = (time.perf_counter() - t0) * 1000

            # Citation validity
            invalid_citations = [hid for hid in hit_ids if hid not in chunk_ids]
            citation_check = CheckResult(
                name="citations_exist",
                passed=len(invalid_citations) == 0,
                detail=f"Invalid citations: {invalid_citations}" if invalid_citations else "All citations valid",
            )
            checks.append(citation_check)

            # Recall@5
            relevant = set(case.relevant_chunk_ids) if case.relevant_chunk_ids else set(case.relevant_doc_ids)
            hits_set = set(hit_ids)
            if case.relevant_chunk_ids:
                recall = len(relevant & hits_set) / len(relevant) if relevant else 0.0
            else:
                hit_doc_ids = {h.split("::")[0] for h in hit_ids}
                recall = len(relevant & hit_doc_ids) / len(relevant) if relevant else 0.0

            recall_check = CheckResult(
                name="recall_at_5",
                passed=recall >= 0.0,  # Actual gate threshold applied by runner
                detail=f"recall@5={recall:.3f}",
                score=recall,
            )
            checks.append(recall_check)

            # Track per-intent recall for metrics
            intent = case.intent or "unknown"
            if intent not in intent_recalls:
                intent_recalls[intent] = {"sum": 0.0, "n": 0}
            intent_recalls[intent]["sum"] += recall
            intent_recalls[intent]["n"] += 1

            # Per-mode recalls
            for mode in ["bm25", "dense", "hybrid"]:
                m_request = type("Request", (), {"query": case.query, "k": 5, "mode": mode, "filters": None})()
                m_response = retriever.retrieve(m_request)
                m_hit_ids = [h.chunk_id for h in m_response.hits]
                if case.relevant_chunk_ids:
                    m_recall = len(set(case.relevant_chunk_ids) & set(m_hit_ids)) / len(case.relevant_chunk_ids) if case.relevant_chunk_ids else 0.0
                else:
                    m_hit_doc_ids = {h.split("::")[0] for h in m_hit_ids}
                    m_recall = len(set(case.relevant_doc_ids) & m_hit_doc_ids) / len(case.relevant_doc_ids) if case.relevant_doc_ids else 0.0
                mode_recalls[mode] += m_recall

            passed = all(c.passed for c in checks)
            trace.set_attributes(
                passed=passed,
                recall_at_5=round(recall, 3),
                n_retrieved=len(hit_ids),
            )
            results.append(CaseResult(
                id=case.id,
                suite="retrieval",
                passed=passed,
                checks=checks,
                retrieved_ids=hit_ids,
                cited_ids=hit_ids,
                latency_ms=latency_ms,
                trace_id=trace.trace_id,
            ))

    n = len(cases)
    metrics = {
        f"recall@5_{mode}": mode_recalls[mode] / n if n else 0.0
        for mode in mode_recalls
    }
    # Per-intent recalls
    for intent, data in intent_recalls.items():
        metrics[f"recall@5_{intent}"] = data["sum"] / data["n"] if data["n"] else 0.0

    sm = SuiteMetrics(
        n=n,
        n_pass=sum(1 for r in results if r.passed),
        n_fail=sum(1 for r in results if not r.passed),
        metrics=metrics,
    )
    return results, sm

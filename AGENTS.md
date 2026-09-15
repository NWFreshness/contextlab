# ContextLab — AI Engineering Teaching System

## What
- **Slice 1 (done)**: Local hybrid retrieval — BM25 + dense (sentence-transformers) + RRF fusion
- **Slice 2 (done)**: Token-budgeted context assembler — priority packing, drop reports, citation integrity
- **Slice 3 (done)**: Evaluation harness — golden suites, deterministic graders, gates, JSON report
- **Slice 4 (done)**: LLM tracer — OpenTelemetry-shaped spans per hop, JSONL store, `trace show`
- **Slice 5 (done)**: Model router — yaml policy (cheap/strong/fallback), `router.decide` spans, dry-run route gate
- **Slice 6 (done)**: Semantic cache — similarity + settings fingerprint + meaning guards, hit/false-hit gates
- **Slice 7 (done)**: Bounded agent orchestrator — `for step in range(max_steps)`, scripted + llm drivers, `orch.run`/`orch.step` spans, trajectory JSON per run

## Run / Verify
```bash
# Slice 1 — retrieval
python -m contextlab.ingest
python -m contextlab.retrieve --query "E-4471" --k 5 --mode hybrid
pytest -q

# Slice 2 — assembly
python -m contextlab assemble --query "what does E-4471 mean" --budget 800 --k 8
pytest tests/test_assemble.py -q

# Slice 3 — eval harness
python -m contextlab.evals run --suite all --offline

# Slice 4 — tracer
pytest tests/test_trace.py -q
python -m contextlab.trace show --last
python -m contextlab.trace show --case r001

# Slice 5 — router
pytest tests/test_router.py -q
python -m contextlab.route --query "what does E-4471 mean" --dry-run
python -m contextlab.evals run --suite router --offline
python -m contextlab assemble --query "what does E-4471 mean" --budget 800 --generate

# Slice 6 — cache
pytest tests/test_cache.py -q
python -m contextlab.cache lookup --query "What does E-4471 mean?"
python -m contextlab.cache seed --from-evals
python -m contextlab.evals run --suite cache --offline
python -m contextlab.cache stats

# Slice 7 — orchestrator
pytest tests/test_orch.py -q
python -m contextlab.orch run --query "what does E-4471 mean" --driver script
python -m contextlab.trace show --last   # orch.run + orch.step tree

# Root verify (--suite all runs retrieval, assembly, answer, router, cache)
pytest -q && python -m contextlab.evals run --suite all --offline && python -m contextlab.trace show --last
```

## Features (feature_list.json)
Retrieval: feat-r1 … feat-r7 (done)
Assembly: feat-a1 … feat-a5 (done)
Eval: feat-e1 … feat-e5 (done)
Trace: feat-t1 … feat-t5 (done)
Router: feat-m1 … feat-m5 (done)
Cache: feat-c1 … feat-c5 (done)
Orchestrator: feat-o1 … feat-o5 (done)

## Hard Bans
- No LangChain / LlamaIndex / Haystack
- No hosted vector DB
- No query-rewrite LLM in Slice 1
- No LLM calls to choose what to drop in Slice 2
- No silent truncation (must drop whole blocks)
- No fake recall numbers
- Do not delete or shrink golden cases
- No hosted observability SaaS as a required dependency (Slice 4: stdlib + pydantic, JSONL store)
- No LLM-as-router unless `policy: llm` in config/router.yaml, and the rules path keeps its tests
- No invented prices — unknown price is `null`, and cost is scored as skipped
- The router must not steer retrieval (bm25/dense/hybrid stays a retrieve flag)
- No cache that only reports hit rate: the false-hit gate is the point
- Never serve a cache hit across a corpus ingest (corpus_version is in the fingerprint)
- No LLM-as-judge for cache hits — guards are deterministic token logic
- No Redis/memcached as a required dependency (local JSONL store)
- Do not relax the must-miss gate to make a threshold look good
- No `while True` without a tested cap (Slice 7: `for step in range(max_steps)`)
- No LangGraph / LangChain / CrewAI / AutoGen — Slice 7 is a state machine, not a framework
- No shell / HTTP / file-write tool in Slice 7 (the tool list lives in `config/orchestrator.yaml`)

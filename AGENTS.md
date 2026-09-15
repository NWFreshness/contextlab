# ContextLab — AI Engineering Teaching System

## What
- **Slice 1 (done)**: Local hybrid retrieval — BM25 + dense (sentence-transformers) + RRF fusion
- **Slice 2 (done)**: Token-budgeted context assembler — priority packing, drop reports, citation integrity
- **Slice 3 (done)**: Evaluation harness — golden suites, deterministic graders, gates, JSON report
- **Slice 4 (done)**: LLM tracer — OpenTelemetry-shaped spans per hop, JSONL store, `trace show`
- **Slice 5 (done)**: Model router — yaml policy (cheap/strong/fallback), `router.decide` spans, dry-run route gate
- **Slice 6 (done)**: Semantic cache — similarity + settings fingerprint + meaning guards, hit/false-hit gates
- **Slice 7 (done)**: Bounded agent orchestrator — `for step in range(max_steps)`, scripted + llm drivers, `orch.run`/`orch.step` spans, trajectory JSON per run
- **Slice 8 (done)**: Sandboxed tool executor — policy gate (allowlist, schema, path jail, env allowlist), `python_calc` AST eval in subprocess, wall-clock timeout, `tool.exec` spans, `prlimit` opt-in
- **Slice 9 (done)**: Trajectory eval suite — constraint graders over `Trajectory` (stop reason, step cap, tools, citations, observations), `evals/trajectory_golden.jsonl` (t001–t012), gates in `evals/gates.yaml`

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

# Slice 8 — sandbox
pytest tests/test_sandbox.py -q
python -m contextlab.orch run --query "what is 2*(3+4)" --driver script   # python_calc
python -m contextlab.orch run --query "what does E-4471 mean" --driver script  # retrieve + read_chunk
python -m contextlab.trace show --last   # orch.run / orch.step / tool.exec tree

# Slice 9 — trajectory evals
pytest tests/test_traj_evals.py -q
python -m contextlab.evals run --suite trajectory --offline
python -m contextlab.trace show --case t001

# Root verify (--suite all runs retrieval, assembly, answer, router, cache, trajectory)
pytest -q && python -m contextlab.evals run --suite all --offline
```

## Features (feature_list.json)
Retrieval: feat-r1 … feat-r7 (done)
Assembly: feat-a1 … feat-a5 (done)
Eval: feat-e1 … feat-e5 (done)
Trace: feat-t1 … feat-t5 (done)
Router: feat-m1 … feat-m5 (done)
Cache: feat-c1 … feat-c5 (done)
Orchestrator: feat-o1 … feat-o5 (done)
Sandbox: feat-s1 … feat-s5 (done)
Trajectory evals: feat-j1 … feat-j5 (done)

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
- No general `shell` or `python -c` tool (Slice 8 hard ban); `python_calc` is the only sandboxed tool
- No `eval()` on raw strings in Slice 8 — only AST whitelisted arithmetic
- No inheriting the parent env wholesale — `env_allowlist` in `config/sandbox.yaml` is the only env a child sees
- No Docker / required external runtime — Slice 8's subprocess backend must work without it; `prlimit` is opt-in
- No disabling the timeout to make a test pass (Slice 8 hard ban)
- No exact-step-dump goldens for the trajectory suite — constraint grading only (Slice 9)
- No LLM judge on tool order / step lists — graders are deterministic (Slice 9)
- Do not tune `evals/trajectory_golden.jsonl` to hide the recorded mutation (Slice 9)
- Do not mark Slice 9 done on handmade fixtures alone — every case must run through `orch.orchestrate`

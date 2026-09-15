# ContextLab — Local Retrieval, Context Assembly, Tracing, Routing, Cache, Agents, Sandbox, Trajectory Evals

A from-scratch teaching system for AI engineering. Nine slices, no
frameworks: retrieval → pack → eval → trace → route → cache → bounded loop
→ sandbox → trajectory gates.

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Ingest corpus
python -m contextlab.ingest

# Query
python -m contextlab.retrieve --query "what does E-4471 mean" --k 5 --mode hybrid

# Assemble context
python -m contextlab assemble --query "what does E-4471 mean" --budget 800 --k 8

# Run the bounded orchestrator (script driver, offline)
python -m contextlab.orch run --query "what does E-4471 mean" --driver script

# Run tests
pytest -q

# Evaluate everything (retrieval, assembly, answer, router, cache, trajectory)
python -m contextlab.evals run --suite all --offline
```

## Architecture

```
Slice 1: corpus/*.md → chunking → chunks.jsonl → bm25_index + dense_index → retrieve → Hit list
Slice 2: assemble(request) → AssembledContext → prompt (token-budgeted, drop-report)
Slice 3: evals/*.jsonl → suites → graders → gates → artifacts/eval_report.json
Slice 4: every hop → spans (trace_id, parent_span_id, attributes) → data/traces/YYYYMMDD.jsonl
Slice 5: route(request, signals) → config/router.yaml → Decision (cheap / strong / fallback)
Slice 6: cache.lookup(query, fingerprint) → similarity + meaning guards → HIT / MISS → data/cache.jsonl
Slice 7: orch.run(query) → for step in range(max_steps) → orch.step → Action → ToolResult → artifacts/trajectories/<id>.json
Slice 8: tool.exec → SandboxPolicy.check → subprocess.run(cwd, env_allowlist, timeout) → ToolResult | ToolError
Slice 9: evals/trajectory_golden.jsonl → orch.run(driver=script) → graders_trajectory.grade → gates
```

### Slice 1 — Retrieval
- **BM25**: Pure-Python Okapi BM25 with tiktoken cl100k_base
- **Dense**: sentence-transformers all-MiniLM-L6-v2 embeddings
- **Fusion**: Reciprocal Rank Fusion (RRF k=60)

### Slice 2 — Context Assembler
- Priority packing (query → system → must_keep tools → retrieval → memory → optional tools → extra)
- Drop report shows exactly what was dropped and why
- Citation integrity: every `[chunk_id]` in prompt resolves to a real chunk
- Token counting: tiktoken cl100k_base (same as chunking)

### Slice 3 — Eval Harness
- 3 golden suites (retrieval / assembly / answer), deterministic graders, gate thresholds in `evals/gates.yaml`
- Report schema includes per-case `trace_id` (Slice 4)

### Slice 4 — Tracer
- Hop spans: `ingest`, `retrieve.bm25`, `retrieve.dense`, `retrieve.fuse`, `assemble.pack`, `generate`, `eval.case`
- Records mirror the OTel span model so an OTel exporter is a thin adapter; no collector required
- JSONL by default, in-memory exporter in tests; secrets redacted on export

### Slice 5 — Model Router
- Deterministic policy in `config/router.yaml`: first matching rule wins, three routes only (cheap / strong / fallback)
- Signals: identifier regex, keyword `intent_hint` (negative / version / table / paraphrase / lexical / other), `n_retrieved`, `conflict` (memory + retrieved hits both present)
- Every decision is a `router.decide` span; the eval suite grades route accuracy on `evals/router_golden.jsonl`
- Dry-run by default: `python -m contextlab.route --query ... --dry-run` decides without HTTP

### Slice 6 — Semantic Cache
- A hit needs a similar query **and** an identical fingerprint (corpus version, retrieve mode, budget, system prompt, route/model)
- Meaning guards (identifiers, numbers, capitalized entities, version words) block near-duplicates that must not share an answer
- Local JSONL store; `cache.lookup` / `cache.write` spans on every attempt; `evals/cache_golden.jsonl` gates hit rate and false-hit rate
- The measured bands overlap (a must-miss pair scores 0.986, same-answer paraphrases 0.54–0.86), so the guards — not the threshold — carry correctness

### Slice 7 — Bounded Agent Orchestrator
- `for step in range(max_steps)` — never `while True`; hitting the cap is a successful machine stop (`stop_reason=max_steps`)
- State is a pydantic model (`State.model_dump()` after every step); the action list replays
- Scripted driver by default (offline, deterministic); `driver: llm` is a structured-output stub that fails closed without a client
- `orch.run` root span with one `orch.step` child per iteration; `assemble.pack` runs after the loop, before the trajectory is written
- Stop reasons: `done` | `max_steps` | `unknown_tool` | `budget` | `error`; the trajectory file is written on every exit

### Slice 8 — Sandboxed Tool Executor
- **Policy gate (always)**: tool allowlist, closed `args_schema`, path jail (after `..` resolution), env allowlist
- **Containment (best-effort)**: `subprocess.run` with a fresh temp `cwd`, allowlisted `env`, `stdin=DEVNULL`, wall-clock `timeout`, output cap
- `python_calc` is the one sandboxed tool: an AST walk over numbers and `+ - * / // % **`. No `eval()` on raw strings
- `retrieve` / `read_chunk` stay in-process (Slice 1 library calls) — `sandbox: false` per tool in `config/sandbox.yaml`
- Every dispatch is a `tool.exec` span: `tool`, `sandbox`, `backend`, `timeout_s`, `exit_code`, `code`
- `prlimit` is opt-in (`cpu_s`); the memory cap is deliberately not applied (Python startup imports blow past 256 MiB on OpenBLAS)

### Slice 9 — Trajectory Eval Suite
- **Constraint goldens, not step dumps**: a row asserts what must be true (stop reason, step cap, required/forbidden tools, tool order, citations, observation substrings)
- 12 graders over a `Trajectory`; every one deterministic — no LLM judge on step lists
- Every case runs through `orch.orchestrate`, so the sandbox and tracer are actually exercised
- `evals/gates.yaml` → `trajectory: {min_pass_rate: 0.85, max_step_cap_violations: 0, max_unknown_tool_on_happy_path: 0}`
- Recorded mutation: `max_steps: 1` flips the suite from 12/12 (pass_rate 1.000) to 1/12 (0.083, gate FAIL)

## Configuration

| File | What it controls |
|------|-----------------|
| `config/retrieval.yaml` | chunk size, embedding model, RRF parameters |
| `config/assembly.yaml` | reserve budgets, citation overhead |
| `config/trace.yaml` | tracing on/off, trace dir, redact keys, include_prompt |
| `config/router.yaml` | policies, models, prices, intent keywords, rules |
| `config/cache.yaml` | threshold, store path, fingerprint requirement, guards |
| `config/orchestrator.yaml` | max_steps, driver, registered tools |
| `config/sandbox.yaml` | backend, timeout, output cap, env/path allowlists, per-tool sandbox flag |
| `prompts/system_default.md` | system prompt |
| `.env` | optional: EMBEDDING_MODEL, OPENAI_API_KEY |

## CLI Commands

```bash
# Retrieval
python -m contextlab.retrieve --query "..." --k 5 --mode hybrid

# Assembly (Slice 2)
python -m contextlab assemble --query "..." --budget 800 --k 8 [--memory-file memory.jsonl]

# Both via module
python -m contextlab retrieve --query "..."
python -m contextlab assemble --query "..."

# Eval harness (Slice 3)
python -m contextlab.evals run --suite all --offline

# Traces (Slice 4)
python -m contextlab.trace show --last
python -m contextlab.trace show --trace <trace_id>
python -m contextlab.trace show --case r001
python -m contextlab.trace list

# Router (Slice 5)
python -m contextlab.route --query "what does E-4471 mean" --dry-run
python -m contextlab.route --query "how long can I return a headset" --dry-run
python -m contextlab.evals run --suite router --offline

# Cache (Slice 6)
python -m contextlab.cache lookup --query "What does E-4471 mean?"
python -m contextlab.cache seed --from-evals
python -m contextlab.cache stats
python -m contextlab.cache clear --yes
python -m contextlab.evals run --suite cache --offline

# Orchestrator (Slice 7)
python -m contextlab.orch run --query "what does E-4471 mean" --driver script
python -m contextlab.orch run --query "what is 2*(3+4)" --driver script       # python_calc
python -m contextlab.orch run --query "what does E-4471 mean" --max-steps 6
python -m contextlab.trace show --last   # orch.run / orch.step tree

# Sandbox (Slice 8)
pytest tests/test_sandbox.py -q
# Backend is config/sandbox.yaml: backend (subprocess|inprocess), timeout_s,
# env_allowlist, path_allowlist, and per-tool {sandbox: true|false}
python -m contextlab.trace show --last   # tool.exec spans

# Trajectory evals (Slice 9)
pytest tests/test_traj_evals.py -q
python -m contextlab.evals run --suite trajectory --offline
python -m contextlab.evals run --suite all --offline
python -m contextlab.trace show --case t001
```

## Cache — similarity is not identity (Slice 6)

```bash
$ python -m contextlab.cache lookup --query "What does E-4472 mean?"
=== Cache lookup ===
query          What does E-4472 mean?
fingerprint    corpus=00f9ec129d629061|mode=hybrid|budget=800|system=2ef4e04cde76d303|route=cheap|model=gpt-4o-mini
threshold      0.79  (require_fingerprint=true)
result         MISS  reason=guard:identifier_mismatch e-4472  score=0.9633  cache_id=-
detail         similar to c12bff27ad5e34e6 ('What does E-4471 mean?') but the guard fired
```

A hit requires **all** of:

1. **Similarity** — cosine ≥ `threshold` against a stored query (or an exact
   normalized-query match, which needs no embedding).
2. **An identical fingerprint** — `corpus_version` (hash of `data/chunks.jsonl`),
   `retrieve_mode`, `budget_tokens`, `system_hash`, and `route`/`model` when the
   router ran. A re-ingest, a different budget or a different routed model is a
   different request, even for byte-identical text.
3. **No meaning guard firing** — identifiers (`E-4471` ≠ `E-4472`), numbers, mid-sentence
   capitalized entities (`West Coast` ≠ `East Coast`), and version words
   (`old/previous/v1` vs `current/now/v3`). Guards are deterministic token logic;
   no LLM judges a cache hit, and a guard can only cost a hit, never cause a wrong one.

Why guards and not just a stricter threshold: on the golden set the bands are
**inverted** — the closest must-miss pair (`West Coast` vs `East Coast`) scores
0.986 while same-answer paraphrases score 0.79–0.86. No threshold can separate
them, so the threshold is a similarity *floor* (0.79) and the guards plus the
fingerprint carry correctness. `evals/gates.yaml`:

```yaml
cache:
  min_should_hit_rate: 0.80   # measured 0.875 (7/8)
  max_false_hit_rate: 0.00    # measured 0.000 (0/6)
```

Configuration lives in `config/cache.yaml` (store path, threshold, fingerprint
requirement, guards). The store is `data/cache.jsonl` — one entry per line, local,
no server. Entries hold the answer plus assembled citation ids, so a hit skips
generation; retrieval and packing still run (milliseconds, no tokens) because the
fingerprint needs the routed model.

## Routing — which model serves this request (Slice 5)

```bash
$ python -m contextlab.route --query "what does E-4471 mean" --dry-run
=== Route decision ===
query          what does E-4471 mean
route          cheap
model          gpt-4o-mini
reason         rule[2] has_identifier=true -> cheap
signals        intent_hint=lexical has_identifier=true n_retrieved=5 conflict=false budget_tokens=800
prompt_tokens  299
dry_run        true

[dry-run] no model was called — pass --generate to call the routed model
```

Policy lives in `config/router.yaml` — never in CLI if-statements. Rules are
evaluated in order, first match wins:

| rule | when | route |
|------|------|-------|
| 0 | `intent_hint in [version, table]` **or** a memory-vs-retrieval `conflict` | strong |
| 1 | `intent_hint == negative` (out of corpus, or a retrieval that returned nothing) | cheap |
| 2 | `has_identifier` (E-4471, POL-REF-30) | cheap |
| 3 | `intent_hint == paraphrase` | strong |
| — | nothing matched | `default_route` (cheap) |

To change the policy, edit the yaml: add a rule, edit an intent keyword list, or
point `models:`/`prices_usd_per_1m:` at the models you actually pay for. Unknown
`when` keys, routes missing from `models:`, and unknown policy names are config
errors — a dead rule is never silently ignored.

- **Models.** `cheap` is `gpt-4o-mini` (what Slice 3 generate used); `strong` and
  `fallback` ship as `PLACEHOLDER_*` names. A live call refuses a placeholder
  **primary** model instead of sending a 404 to a paid API; a placeholder
  **fallback** just means "no fallback configured" (`fallback_configured: false`
  on the span) and never blocks a working primary.
- **Prices.** `prices_usd_per_1m` entries are `null` until you fill them in. `null`
  means "unknown": `cost_usd_estimate` is then omitted from spans and reports —
  never written as `0.0`. A number is a flat per-1M rate; `{input:, output:}` splits it.
- **Fallback.** `fallback_on_error: true` retries once with `models.fallback` when
  the primary raises. The `generate` span stays `ok` with `fallback_used: true` and
  `primary_error` recorded; if both fail the error propagates and the span is `error`.
- **Gate.** `evals/gates.yaml` → `router.min_accuracy: 0.85` over
  `evals/router_golden.jsonl` (19 labeled cases, dry-run). The gate reads
  `route_accuracy` only; a case also fails if its `intent` label stops matching the
  policy, so a keyword edit cannot silently re-label the goldens.
- **Not here:** the router does not choose bm25 vs dense (`--mode` on retrieve does),
  and there is no cache yet.

## Tracing — debugging a failed eval case (Slice 4)

Every CLI hop writes spans; nothing is sent to a hosted observability service.

```bash
# 1. run an assembler and note the printed trace id
python -m contextlab assemble --query "what does E-4471 mean" --budget 250
#   [trace] 9f2c… — python -m contextlab.trace show --trace 9f2c…

# 2. or reopen the newest trace in the store
python -m contextlab.trace show --last

# 3. after an eval run, reopen the exact case that failed
python -m contextlab.evals run --suite all --offline
python -m contextlab.trace show --case r001
```

The tree prints each hop's duration, status, and the ids you need to explain a bad
answer: `chunk_ids` / `doc_ids` on the retrieve span, `kept_ids` / `dropped_ids` and
`prompt_tokens` on `assemble.pack`, `model` / token counts on `generate`.

Spans store metadata, not prompts: `prompt_sha256`, `prompt_chars` and an 80-char
`prompt_preview`. Set `include_prompt: true` in `config/trace.yaml` only when you
deliberately want full prompts on disk.

A hop that closes without its required attributes is flagged in the tree
(`MISSING: [...]`) and in `attributes.missing_context`; an exception leaves the span
`error` and the trace is still exported (`show` prints the error text).

Redaction: attribute keys matching `redact_keys` (case-insensitive, matched segment by
segment) become `"***"`, and values are scrubbed for `sk-…` / `Bearer …` /
`key=secret` patterns. `prompt_tokens` / `input_tokens` are token *counts* and stay
readable.

Env overrides: `CONTEXTLAB_TRACE_ENABLED=0|1`, `CONTEXTLAB_TRACE_DIR=<path>`,
`CONTEXTLAB_TRACE_REDACT_KEYS=a,b`, `CONTEXTLAB_TRACE_INCLUDE_PROMPT=0|1`.

## Trajectory evals — grading the path, not just the answer (Slice 9)

```bash
$ python -m contextlab.evals run --suite trajectory --offline
=== Eval Summary ===
  trajectory: 12/12 passed
    pass_rate: 1.000
    case_types: {'happy': 9, 'max_steps': 1, 'unknown_tool': 1, 'policy': 1}
    step_cap_violations: 0
    unknown_tool_on_happy_path: 0
Gates:
  [PASS] gate_trajectory_pass_rate: pass_rate=1.000 >= 0.85
  [PASS] gate_trajectory_step_caps: step_cap_violations=0 <= max=0
  [PASS] gate_trajectory_unknown_tool_happy: unknown_tool_on_happy_path=0 <= max=0

=== EVAL PASSED ===
```

Each golden row lists **constraints**, not an expected step dump. A step
dump breaks the moment `retrieve` returns one extra hit; a constraint
(`stop_reason=done`, `required_tools=[retrieve]`, `observed 14`) is stable
and still catches real regressions:

```json
{"id": "t005", "query": "what is 2*(3+4)", "case_type": "happy",
 "expected_stop_reasons": ["done"], "required_tools": ["python_calc"],
 "forbidden_tools": ["retrieve"], "required_observation_substrings": ["14"]}
```

Twelve graders run per case — `stop_reason`, `step_cap`, `required_tools`,
`forbidden_tools`, `tool_order`, `citations`, `obs_contains`, `obs_forbids`,
`final_contains`, `final_forbids`, `trace_present`, `prompt_tokens` — and
every one is a deterministic string/set comparison. There is no LLM judge
on a step list.

Two cases use a named fixture policy (`policy_name`) instead of the
production script policy: `always_retrieve` (drives the `max_steps` cap
with `settings_override: {"max_steps": 1}`) and `unknown_tool` (emits a
tool name outside the allowlist). Those fixtures live in
`orch/script_policy.FIXTURE_POLICIES`; the eval runner has no hidden ifs.

**The mutation that proves the gate works.** Editing
`config/orchestrator.yaml` to `max_steps: 1`:

| | default (`max_steps: 6`) | mutation (`max_steps: 1`) |
|---|---|---|
| passed | 12/12 | **1/12** |
| pass_rate | 1.000 | **0.083** |
| `gate_trajectory_pass_rate` | PASS | **FAIL** (exit 1) |

Every happy-path case flips while `stop_reason` stays `done` — the
orchestrator still stops "successfully", it just never runs a tool. Only
the tool-presence and observation constraints catch that. A
final-answer-only eval scores the mutation 12/12.

## Next Slice

Stop. ContextLab 01–09 is a complete teaching system: retrieve → pack →
eval → trace → route → cache → bounded loop → sandbox → trajectory gates.

Proposed later work, only if a named failure justifies it: prompt
registry, streaming proxy, MCP, durable checkpoints.

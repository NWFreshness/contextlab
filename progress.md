# ContextLab — Slice 6 Progress (Semantic Cache)

## Current Verified State

```
Date: 2026-09-14
Status: COMPLETE ✓
should_hit_rate 0.875 (7/8, gate >= 0.80)   false_hit_rate 0.000 (0/6, gate == 0.00)
Slices 1-5 unchanged: retrieval recall@5_hybrid 0.9167, assembly 7/7, answer 13/14 (ans003), router 19/19
```

### Verified Commands
| Command | Exit | Result |
|---------|------|--------|
| `pytest -q` | 0 | 170 passed (128 from Slices 1-5 + 42 cache) |
| `pytest tests/test_cache.py -q` | 0 | 42 passed |
| `python -m contextlab.cache lookup --query "What does E-4471 mean?"` | 0 | prints HIT/MISS, score, cache_id (empty store → `MISS reason=miss`) |
| `python -m contextlab.cache seed --from-evals` | 0 | 14 entries → `data/cache.jsonl` |
| `python -m contextlab.cache stats` | 0 | entries, corpus versions, modes, routes |
| `python -m contextlab.evals run --suite cache --offline` | 0 | 13/14 cases, both cache gates PASS |
| `python -m contextlab.evals run --suite all --offline` | 0 | 10/10 gates PASS across retrieval, assembly, answer, router, cache |
| `python -m contextlab.trace show --last` | 0 | `cache.lookup` + `cache.write` under `eval.case` |
| `python -m contextlab assemble --query ... --generate` | 0 | `[router] …` then `[cache] miss (…)` then `[generate skipped] …` |

### Done Criteria
- [x] `python -m contextlab.cache lookup --query "..."` prints hit/miss, score, cached id
- [x] The generate path consults the cache before generate and writes on a miss after a successful generate
      (`cache/answer.py`; offline tests seed from fixtures, no live LLM)
- [x] Cache suite gated in `evals/gates.yaml` (`min_should_hit_rate: 0.80`, `max_false_hit_rate: 0.00`)
- [x] Mutation recorded: `threshold: 0.99` → should_hit 0.250 (FAIL); `require_fingerprint: false` →
      false_hit 0.333 (FAIL, via the real eval CLI); guards off at 0.50 → false_hit 0.667 (FAIL)
- [x] `trace show --last` shows `cache.lookup`
- [x] Slice 3 + 5 offline evals still pass (all 10 gates PASS, recall unchanged)

---

## Session: 2026-09-14

### What was built
- `config/cache.yaml` — store path, threshold, fingerprint requirement, exact layer, meaning guards
- `src/contextlab/cache/fingerprint.py` — `CacheFingerprint` (corpus_version / retrieve_mode / budget_tokens /
  system_hash / route / model), query normalization, corpus + system hashing
- `src/contextlab/cache/store.py` — `CacheEntry`, `CacheStore` (JSONL, one entry per line), stats
- `src/contextlab/cache/lookup.py` — `Cache.lookup` / `Cache.write` with `cache.lookup` / `cache.write` spans,
  cosine over stored query embeddings, exact layer, guards
- `src/contextlab/cache/config.py` — config loading + validation, `CONTEXTLAB_CACHE_PATH` / `_ENABLED` overrides
- `src/contextlab/cache/answer.py` — the one place the cache is wired in: assemble → route → lookup →
  (hit: stored answer) / (miss: generate → write)
- `src/contextlab/cache/__main__.py` — `lookup` / `seed --from-evals` / `stats` / `clear --yes`
- `evals/cache_golden.jsonl` (14 rows), `src/contextlab/evals/suites_cache.py`, `tests/test_cache.py` (42 tests)
- Wiring: `evals/runner.py` (suite + gates, included in `--suite all`), `evals/__main__.py`,
  `evals/gates.yaml`, `trace/types.py` (`cache.lookup`, `cache.write` contracts), `cli.py` + `route.py`
  (both answer paths now route → cache → generate)
- Docs: AGENTS.md, README (cache section), feature_list.json (feat-c1 … feat-c5)

### The measurement that shaped the design

Cosine similarity between every probe and its seed (all-MiniLM-L6-v2, the dense-retrieval model):

| case | type | cosine | what actually stops it |
|------|------|--------|------------------------|
| c001 | should_hit | 0.7945 | — (hit) |
| c002 | should_hit | 0.7999 | — (hit) |
| c003 | should_hit | 0.8235 | — (hit) |
| c004 | should_hit | 0.8405 | — (hit) |
| c005 | should_hit | 0.8636 | — (hit) |
| c006 | should_hit | 0.9523 | — (hit, exact layer) |
| c007 | should_hit | 1.0000 | — (hit, exact layer) |
| c013 | should_hit | 0.5406 | threshold (miss, kept in the golden) |
| c008 | must_miss | 0.7332 | guard:version_conflict |
| c009 | must_miss | 0.9633 | guard:identifier_mismatch |
| c010 | must_miss | **0.9860** | guard:entity_mismatch (east) |
| c011 | must_miss | 1.0000 | fingerprint (retrieve_mode) |
| c012 | must_miss | 1.0000 | fingerprint (budget) |
| c014 | must_miss | 0.6556 | guard:entity_mismatch + threshold |

**The bands are inverted.** The closest must-miss pair (West Coast vs East Coast, different answer) scores
0.986 while same-answer paraphrases score 0.79-0.86. No threshold can separate them: at 0.99 the cache hits
2/8; at 0.79 it hits 7/8 but would also hit c010 if no guard existed. So the threshold is a similarity *floor*
and **the guards plus the fingerprint carry correctness** — which is what the brief asked for when it said to
pull the "required token overlap check" lever. Recorded in config/cache.yaml comments and README.

### Cache golden result (shipped)
```
  cache: 13/14 passed
    should_hit_rate: 0.875     (gate >= 0.80  PASS)   7/8
    false_hit_rate: 0.000      (gate == 0.00  PASS)   0/6
    min_hit_score: 0.795       max_miss_score: 0.986  <- printed every run
    threshold: 0.790           require_fingerprint: True
    reasons: {semantic: 5, exact: 2, guard:version_conflict: 1, guard:identifier_mismatch: 1,
              guard:entity_mismatch: 2, fingerprint_mismatch: 2, miss: 1}
```
Distribution: 8 should_hit (5 paraphrases + 2 exact repeats + 1 deliberately hard paraphrase at 0.54) and
6 must_miss (old-vs-current refund, E-4471-vs-E-4472, West-vs-East Coast, Starter-vs-Professional plan,
same question different retrieve_mode, same question different budget). c013 (the 0.54 paraphrase) is a case,
not a gap: it stays in the golden even though no threshold near the paraphrase band can hit it — a hard case
belongs in the set, not out of it.

### Real dump — `cache lookup` on a seeded store

```
$ python -m contextlab.cache seed --from-evals
  seeded c014   57fd21ce9d0d957a  How much does the Starter plan cost?
14 entries -> /home/tylermayfield/Documents/contextlab/data/cache.jsonl

$ python -m contextlab.cache lookup --query "What does E-4471 mean?"
result         HIT  reason=exact  score=-  cache_id=c12bff27ad5e34e6
detail         normalized query match
cached_query   What does E-4471 mean?
answer         E-4471 means the inventory reservation expired.

$ python -m contextlab.cache lookup --query "Meaning of error code E-4471"
result         HIT  reason=semantic  score=0.7945  cache_id=c12bff27ad5e34e6
detail         cosine 0.7945 >= 0.79

$ python -m contextlab.cache lookup --query "What does E-4472 mean?"
result         MISS  reason=guard:identifier_mismatch e-4472  score=0.9633  cache_id=-
detail         similar to c12bff27ad5e34e6 ('What does E-4471 mean?') but the guard fired

$ python -m contextlab.cache lookup --query "What was the old refund window before the policy change?"
result         MISS  reason=guard:version_conflict  score=0.7332  cache_id=-
detail         similar to 33965b16167cf7f1 ('What is the current refund window?') but the guard fired
```

### Real dump — `trace show --case c014` (trimmed)

```
=== Trace 6d6c581656678ddd6cb27af269780303 ===
root eval.case   13.3 ms   spans 3 (0 in error)
span tree
eval.case           13.3 ms  ok
                    case_id="c014" suite="cache" query="How much does the Professional plan cost?" passed=true
                    case_type="must_miss" expect_hit=false hit=false reason="guard:entity_mismatch professional"
                    score=0.6555719541463411 cache_id=null threshold=0.79
  cache.write         7.6 ms  ok
                    query_chars=36 cache_id="6f38f59ba00b2d1d" answer_chars=37 citations=0
                    embedding_dim=384 fingerprint="corpus=00f9ec129d629061|mode=hybrid|budget=800|system=2ef4…" entries=1
  cache.lookup        5.4 ms  ok
                    threshold=0.79 require_fingerprint=true hit=false score=0.6555719541463411
                    reason="guard:entity_mismatch professional" cache_id=null entries_scanned=1
                    detail="similar to 6f38f59ba00b2d1d ('How much does the Starter pl…"
```

### Mutation experiment (recorded, then reverted)

| run | config | should_hit_rate | false_hit_rate | gate result | false-hit cases |
|-----|--------|-----------------|----------------|-------------|-----------------|
| shipped | threshold 0.79, fingerprint on, guards on | **0.875** | **0.000** | PASS | none |
| mutation 1 | `threshold: 0.99` | 0.250 | 0.000 | FAIL (should-hit) | none — a dead cache: only the exact repeats hit |
| mutation 2 | `require_fingerprint: false` (real eval CLI) | 0.875 | **0.333** | FAIL (false-hit) | c011, c012 |
| mutation 3 | `threshold: 0.50` + all guards explicitly off | 1.000 | **0.667** | FAIL (false-hit) | c008, c009, c010, c014 |

Mutation 2 was run through the actual CLI (`python -m contextlab.evals run --suite cache --offline` with
`config/cache.yaml` edited), producing `=== EVAL FAILED ===`; the config was restored and re-verified
(`gate_cache_false_hit: false_hit_rate=0.000 <= 0.0 => PASS`). Mutations 1 and 3 ran the same suite in-process
with variant configs so no repo file was left broken.

Reading of the three: the threshold alone buys nothing (0.99 kills the hit rate without adding safety), the
fingerprint is load-bearing (it is the only thing blocking identical questions with a different budget/mode),
and the guards are what make similarity usable (without them a *lower* threshold happily serves the 14-day
answer for a 30-day question). A cache that reports only a hit rate would have scored mutation 3 as the best
configuration in this table — 1.000 hit rate. That is why the false-hit gate exists.

### Spot checks (from the brief)
1. Seed "current refund window?" → probe "old refund window before the policy change?" → **MISS**
   (`guard:version_conflict`, cosine 0.7332). ✓
2. Seed E-4471 definition → probe "meaning of E-4471" → **HIT** (`semantic`, cosine 0.7945). ✓
3. Same query, budget 800 vs 400 → **MISS** (`fingerprint_mismatch`, details name `budget_tokens`). Unit test
   `test_same_query_different_budget_is_a_miss` plus golden case c012. ✓
4. `rg -i 'sk-|api_key|authorization|password' data/cache.jsonl` → no hits. ✓

### Findings and judgment calls (all written down)
1. **Three files beyond the brief's list, and why.** `cache/config.py` (the config loader had no natural home
   among store/lookup/fingerprint without a circular import), `cache/answer.py` (one shared assemble → route →
   lookup → generate path for both CLIs; duplicating it in `cli.py` and `route.py` would drift), and
   `cache/__main__.py` (required for `python -m contextlab.cache`, exactly like `trace/__main__.py`).
2. **The lookup runs *after* the routing decision**, not before it as the brief's data-flow sketch shows. The
   fingerprint contains route and model, so a hit must know which model would have served the request; routing
   before assemble would use query-only signals (no conflict / n_retrieved) and could fingerprint the wrong
   route. Cost of the choice: a hit still pays for local retrieval and packing (milliseconds, no tokens).
3. **A hit skips generation and the cache write, not the packer's replay.** What is stored is the answer plus
   the assembled citation ids (`ctx.citations`), not the packed prompt, so a hit does not rebuild the prompt.
4. **Guards default ON — an empty `guards: {}` does not disable them.** Each flag has to be turned off
   explicitly; mutation 3's first attempt proved it (an empty dict still fired identifier/entity guards, and
   only the version-word guard went quiet because its groups list was gone). Safe default, recorded here.
5. **`test_route_cli_version_query_routes_strong` (Slice 5) broke when the cache landed**, because the real
   seeded `data/cache.jsonl` made a CLI probe hit the cache instead of reaching generate. That is a genuine
   cross-test coupling, not a flaky test: tests now redirect the store with `CONTEXTLAB_CACHE_PATH` in
   `tests/conftest.py`, mirroring the tracer's `CONTEXTLAB_TRACE_DIR` from Slice 4. Three tests pin the
   override, and the suite's temp-store test now checks the repo path explicitly rather than the resolved one.
6. **Embedding stack:** same library, same model, same cosine as dense retrieval
   (`config/retrieval.yaml: embedding_model`, `cache.yaml: embedding_model: null` inherits it). It is a second
   in-process instance of that model — sharing the dense retriever's instance would mean threading it through
   `assemble()`, which builds its own `Retriever`. Called out here as the slice's "second embedding stack"
   note; there is no second *stack*, just a second handle.
7. **Prompts are not cached.** The payload is `{answer, citations, model, prompt_tokens}`; the packed prompt
   is never stored, so `data/cache.jsonl` holds the question, the answer and the citation ids — nothing about
   the prompt that produced them (checked with the secret scan and by inspecting the entry fields).
8. **Dry-run semantics:** `answer_with_cache(dry_run=True)` still *serves* a cached answer (it costs nothing)
   and writes nothing on a miss; the CLIs use `--generate` to allow a model call. `route --dry-run` now also
   prints the cache verdict, which is how the fingerprint is inspected in practice.

### Not done here (later slices, not started)
- No bounded orchestrator, no sandboxed tools, no trajectory cases. The second trio (tracer, router, cache)
  is closed.

---

## Slice 5 (archived — model router)

`config/router.yaml` rules (first match wins) → routes cheap/strong/fallback; `router.decide` spans on every
decision; `evals/router_golden.jsonl` (19 cases) gated at `min_accuracy: 0.85`; measured **route_accuracy
1.000**, intent_accuracy 1.000 (routes cheap 8 / strong 11). Fallback: retry once with `models.fallback`, span
`ok` with `fallback_used=true` and `primary_error`; both failing propagates. Prices are `null` → no
`cost_usd_estimate` (never a fake 0.0). `pytest tests/test_router.py -k fallback -q` → 5 passed.

## Slices 1-4 (archived, unchanged)
- Slice 1: BM25 + dense + RRF hybrid retrieval; recall@5_hybrid = 0.9167 over 24 golden cases.
- Slice 2: token-budgeted assembler with drop reports and citation integrity.
- Slice 3: eval harness — suites, gates, `artifacts/eval_report.json`.
- Slice 4: tracer — OTel-shaped spans per hop, JSONL store, `trace show`; the refund `--budget 400` dump lives
  in the Slice 4 record (`assemble.pack` → `dropped_ids=[refund_policy_v3::c0000]`).

---

# ContextLab — Slice 7 Progress (Bounded Agent Orchestrator)

## Current Verified State
```
Date: 2026-09-15
Status: COMPLETE ✓
Verified run: trajectory_id=5d95956c5e97b620  stop_reason=done  driver=script  n_steps=3
Slices 1-6 unchanged: 10/10 eval gates PASS, 181 pytest passed.
```

### Done Criteria (from brief §1)
- [x] `python -m contextlab.orch run --query "what does E-4471 mean" --driver script`
      → step log + `stop_reason=done` + 3 steps + trajectory written
- [x] `pytest tests/test_orch.py -q` covers max_steps cap, scripted happy path,
      unknown_tool, inspectable state — 11 tests, all green
- [x] Trajectory written to `artifacts/trajectories/<trajectory_id>.json`
- [x] `trace show --last` shows `orch.run` with `orch.step` children + nested
      `retrieve.hybrid` + `orch.assemble` → `assemble.pack`
- [x] `--suite all --offline` still passes (10/10 gates, recall unchanged)
- [x] `progress.md` includes trajectory id `5d95956c5e97b620` and `stop_reason=done`

### Spot checks (brief §7)
1. `--max-steps 4` on a policy that always calls `retrieve` → `stop_reason=max_steps`,
   `len(steps)==4` (`test_max_steps_cap_fires_on_runaway_policy`).
2. A fixture policy that picks `shell` (unregistered) → `stop_reason=unknown_tool`,
   trajectory file written so Slice 9 can grade the failure path
   (`test_unknown_tool_policy_stops_with_unknown_tool_reason`).
3. After step 0, `state.model_dump()` round-trips through JSON and contains
   the query, the step index, and `hits=[]`
   (`test_state_is_serializable_after_each_step`).

### What was built
- `config/orchestrator.yaml` — `max_steps: 6`, `driver: script`, `tools: [retrieve, read_chunk, finish]`
- `src/contextlab/orch/__init__.py` — public API: `orchestrate`, `ScriptedPolicy`,
  `LlmPolicy`, `InProcessExecutor`, `Action`, `Step`, `State`, `Trajectory`,
  `OrchRequest`, `UnknownToolError`
- `src/contextlab/orch/state.py` — `Action`, `Step`, `State` (with `summary()` for
  the LLM driver), `Trajectory`, `OrchRequest`, `load_orchestrator_config`
  (env overrides: `CONTEXTLAB_ORCH_MAX_STEPS`, `CONTEXTLAB_ORCH_DRIVER`)
- `src/contextlab/orch/action.py` — `Action` pydantic model + `ActionType` constants
- `src/contextlab/orch/executor_inprocess.py` — `Executor` protocol + `InProcessExecutor`
  (tools: `retrieve`, `read_chunk`). Unknown tool → `UnknownToolError` so the
  machine can set `stop_reason=unknown_tool`.
- `src/contextlab/orch/script_policy.py` — table-driven scripted policy; tracks
  already-read chunks so it can't loop on Rule 1 (`_already_read(state, chunk_id)`)
- `src/contextlab/orch/llm_policy.py` — skip-safe stub: returns
  `Action.fail("llm driver not configured")` when `client=None`; one JSON-parse
  retry when a client is provided
- `src/contextlab/orch/machine.py` — the bounded loop (`for step in range(max_steps)`,
  no `while True`); opens `orch.run` root trace, opens `orch.step` per iteration,
  runs `assemble.pack` after the loop, persists the trajectory
- `src/contextlab/orch/run.py` — `python -m contextlab.orch run` CLI; mirrors the
  `route` CLI shape (stop_reason, n_steps, citations, trace id, step log)
- `src/contextlab/orch/__main__.py` — module entry point
- `tests/test_orch.py` — 11 tests: happy path, max_steps cap, unknown_tool,
  state inspectability, executor contracts (read_chunk missing, unknown tool
  raises, duck-typed protocol), policy termination, trajectory JSON round-trip
- `feature_list.json` — `feat-o1` … `feat-o5` (done)
- `AGENTS.md` — Slice 7 listed under "What", run command under "Run / Verify"

### Real run output (`python -m contextlab.orch run --query "what does E-4471 mean" --driver script`)
```
=== Orchestrator run (script) ===
query           what does E-4471 mean
trajectory_id   5d95956c5e97b620
stop_reason     done
n_steps         3
prompt_tokens   141
[trace] (varies per run) — python -m contextlab.trace show --trace …

=== Steps ===
  step= 0 state=observe  type=call_tool  tool=retrieve     reason=no_hits
  step= 1 state=observe  type=call_tool  tool=read_chunk   reason=identifier:E-4471
  step= 2 state=act      type=finish     tool=-            reason=top_hit_extract

=== Final answer ===
If E-4471 persists after retry, escalate to L2 support with: …
```

### Real `trace show --last` tree (trimmed)
```
orch.run            2546.7 ms  ok
                     query="what does E-4471 mean" driver="script" max_steps=6
                     trajectory_id="5d95956c5e97b620" stop_reason="done"
                     n_steps=3 n_citations=0 prompt_tokens=141
  orch.step           2545.0 ms  ok   step=0 state="observe" tool="retrieve"
    retrieve.hybrid        7.7 ms  ok   query=…  k=5  chunk_ids=[incident_runbook::c0004, …]
      retrieve.bm25          0.4 ms  ok   …
      retrieve.dense         7.0 ms  ok   …
      retrieve.fuse          0.1 ms  ok   …
  orch.step              0.3 ms  ok   step=1 state="observe" tool="read_chunk"
  orch.step              0.0 ms  ok   step=2 state="stop"    action_type="finish"
  orch.assemble          1.1 ms  ok   budget_tokens=800  prompt_tokens=141
    assemble.pack          0.1 ms  ok   kept_ids=[read_chunk:incident_runbook::c0004]
                                            dropped_ids=[retrieve:what does E-4471 mean]
```

### Findings and judgment calls (all written down)
1. **The function is `orchestrate`, not `run`.** The brief specifies a
   `src/contextlab/orch/run.py` for the CLI module; importing that file
   binds `run` to the *module* in `contextlab.orch.__init__`. Renaming the
   bounded-loop function to `orchestrate` keeps `from contextlab.orch import
   run` resolving to the CLI module (the intent) while leaving a `run = orchestrate`
   alias in `machine.py` for callers that want the function directly.
2. **Loop termination on Rule 1.** The scripted policy's Rule 1
   (`read_chunk` for the top hit when the query contains an identifier) needs
   to short-circuit on a re-match of the same chunk, otherwise the machine
   consumes the entire cap reading the same chunk repeatedly. `_already_read`
   is a deterministic prefix match on `state.observations[*].tool_id`.
3. **`assemble.pack` is called with `retrieve=False`.** The orchestrator
   already gathered hits via the `retrieve` tool, so re-running Slice 1
   inside Slice 2 would double the work. The packer's tool-result slots are
   still used (`tools=observations`), keeping the citation + drop-report
   behavior intact.
4. **`last_step_action` is a guard, not a finish.** The brief describes the
   cap as a successful stop; a runaway policy that always returns
   `call_tool` is still expected to stop, with `len(steps)==max_steps`. So
   `last_step_action` is allowed to return `call_tool` — the *loop* fires
   the cap, not the policy. The `max_steps` test pins this.
5. **Trajectory file is written on every exit, including failures.** A
   `stop_reason=unknown_tool` run still produces a JSON file at
   `artifacts/trajectories/<id>.json` so Slice 9's trajectory grader can
   inspect the failure path.
6. **Hard ban check.** No `while True`, no LangGraph/LangChain/CrewAI/AutoGen
   imports, no shell/HTTP/write tool in the registered list. The
   `InProcessExecutor` only knows `retrieve` and `read_chunk`; Slice 8
   swaps the implementation behind the same `Executor` protocol without
   touching the machine or the policies.
7. **`n_citations=0` on the root trace.** The orchestrator's pack step
   keeps `read_chunk` observations and drops `retrieve` observations (the
   latter is redundant once the chunk is loaded). Slice 9 should grade
   trajectory `citations` from `assemble.pack.kept_ids`, not from the raw
   retrieval — both paths are visible in the trace tree.

### Not done here (later slices, not started)
- No sandboxed tool executor (Slice 8 replaces `InProcessExecutor`, not the
  `Executor` protocol or the machine).
- No trajectory eval suite (Slice 9 grades `artifacts/trajectories/*.json`).
- The LLM driver is a skip-safe stub — `driver: llm` returns
  `stop_reason=error` until a real structured-output client is wired in.

---

# ContextLab — Slice 8 Progress (Sandboxed Tool Executor)

## Current Verified State
```
Date: 2026-09-15
Status: COMPLETE ✓
Backend: subprocess (verified default; inprocess kept as the unit-test escape hatch)
Verified run: trajectory_id=be24151f521a1831  stop_reason=done  driver=script  n_steps=2
              query="what is 2*(3+4)"  final_answer="14"  sandbox=true  backend=subprocess
Slices 1-7 unchanged: 10/10 eval gates PASS, 215 pytest passed.
```

### Done Criteria (from brief §1)
- [x] `config/sandbox.yaml` exists; `backend`, `timeout_s`, `max_output_bytes`,
      `cpu_s`, `memory_mb`, `env_allowlist`, `path_allowlist`, `tools` are all
      parsed and used at runtime.
- [x] `SandboxedExecutor` implements the Slice 7 `Executor` protocol
      (`run(tool, args) -> ToolResult`); `UnknownToolError` propagation for
      unregistered tools is preserved (the orchestrator's stop_reason wiring
      is unchanged).
- [x] Orchestrator default executor is `SandboxedExecutor` (from `machine.py`).
      `backend: inprocess` is still selectable via `CONTEXTLAB_SANDBOX_BACKEND`
      for unit tests.
- [x] `pytest tests/test_sandbox.py -q` covers timeout (sleep_forever),
      allowlist (extra-args, unknown tool), path jail (BRIEF.md / ../../../etc/passwd),
      env non-leak (OPENAI_API_KEY never appears in child env), and the AST
      calc happy / error paths — 34 tests, all green.
- [x] `trace show --last` after an arithmetic orch run shows `tool.exec` with
      `tool=python_calc, sandbox=true, backend=subprocess, timeout_s=2.0,
      exit_code=0, code=ok`.
- [x] E-4471 still hits the in-process retrieve / read_chunk path
      (`backend=inprocess, sandbox=false`); the existing orch + eval suites
      still pass (215 pytest, 10/10 eval gates).

### Spot checks (brief §7)
1. Timeout fixture (`sleep_forever seconds=10` against `timeout_s=1.0`) returns
   `code=timeout` in ~1.0s (well under the brief's 5s ceiling).
   Verified in `test_timeout_kills_long_running`.
2. Calc rejects name lookups: `evaluate("__import__('os')")` raises `CalcError`
   with `disallowed syntax: Name`. Verified in `test_rejects_name_lookup`.
3. The orchestrator's last `ToolResult` and the worker's captured stdout do
   not contain `sk-test` — the parent's `OPENAI_API_KEY` is dropped by the
   env allowlist (`env_allowlist: [PATH, LANG, LC_ALL]` in
   `config/sandbox.yaml`). Verified in `test_env_non_leak`.

### What was built
- `config/sandbox.yaml` — `backend: subprocess`, `timeout_s: 2.0`,
  `max_output_bytes: 8000`, `cpu_s: null` (opt-in), `memory_mb: 256`,
  `env_allowlist: [PATH, LANG, LC_ALL]`, `path_allowlist: [data/chunks.jsonl,
  data/sandbox_work]`, per-tool `sandbox: true|false` + `args_schema`.
- `config/orchestrator.yaml` — added `python_calc` to the registered tools.
- `src/contextlab/sandbox/__init__.py` — public API: `SandboxLimits`,
  `ToolSandboxPolicy`, `SandboxPolicy`, `ToolError`, `ErrorCode`,
  `SandboxedExecutor`, `evaluate`, `load_sandbox_config`, `has_prlimit`.
- `src/contextlab/sandbox/limits.py` — `SandboxLimits` pydantic model +
  `load_sandbox_config` with env overrides (`CONTEXTLAB_SANDBOX_BACKEND`,
  `CONTEXTLAB_SANDBOX_TIMEOUT_S`, `CONTEXTLAB_SANDBOX_WORKDIR`).
- `src/contextlab/sandbox/policy.py` — `ToolError`, `ErrorCode`,
  `SandboxPolicy` (allowlist, schema, path jail, env allowlist).
- `src/contextlab/sandbox/calc.py` — AST evaluator (used in-process for
  unit tests; the worker has its own copy to avoid loading the package).
- `src/contextlab/sandbox/worker.py` — subprocess worker that does NOT
  import `contextlab` (so it doesn't pull in sentence-transformers /
  torch / OpenBLAS). Handles `python_calc` plus three test fixtures
  (`sleep_forever`, `echo_env`, `read_secret`).
- `src/contextlab/sandbox/executor.py` — `SandboxedExecutor`:
  `SandboxPolicy.check` → in-process OR `subprocess.run` → ToolResult;
  emits `tool.exec` span with the brief's attribute set.
- `src/contextlab/orch/machine.py` — defaults to `SandboxedExecutor`
  (orchestrator's wiring is otherwise unchanged).
- `src/contextlab/orch/script_policy.py` — Rule -1 routes pure-arithmetic
  queries to `python_calc`; reads `state.observations` so the second step
  finishes with the calc result instead of falling through to retrieve.
- `src/contextlab/trace/types.py` — `tool.exec` registered in
  `REQUIRED_ATTRIBUTES` (tool, backend, timeout_s).
- `tests/test_sandbox.py` — 34 tests: 10 AST calc, 4 policy gate, 11
  subprocess executor, 2 workdir cleanup, 1 prlimit detection, 2
  orchestrator integration, 1 trace contract, 3 config loader.
- `feature_list.json` — `feat-s1` … `feat-s5` (done).
- `AGENTS.md` — Slice 8 listed under "What", run commands under
  "Run / Verify", hard bans updated.

### Real run output (`python -m contextlab.orch run --query "what is 2*(3+4)" --driver script`)
```
=== Orchestrator run (script) ===
query           what is 2*(3+4)
trajectory_id   be24151f521a1831
stop_reason     done
n_steps         2
prompt_tokens   61
[trace] (varies per run)

=== Steps ===
  step= 0 state=observe  type=call_tool  tool=python_calc  reason=arithmetic
  step= 1 state=act      type=finish     tool=-            reason=arithmetic_result

=== Final answer ===
14
```

### Real `trace show --last` tree (trimmed)
```
orch.run            <duration> ms  ok
                     query="what is 2*(3+4)" driver="script" max_steps=4
                     trajectory_id="be24151f521a1831" stop_reason="done" n_steps=2
  orch.step           <duration> ms  ok   step=0 state="observe" tool="python_calc"
    tool.exec             ~40 ms   ok   tool="python_calc" sandbox=true
                                            backend="subprocess" timeout_s=2.0
                                            args_keys=[expression] exit_code=0 code="ok"
  orch.step            <duration> ms  ok   step=1 state="stop" action_type="finish"
  orch.assemble        <duration> ms  ok   budget_tokens=400 prompt_tokens=61
    assemble.pack          <duration> ms  ok   kept_ids=[python_calc:ok]
                                              dropped_ids=[]
```

### Findings and judgment calls (all written down)
1. **The worker doesn't import `contextlab`.** `python -m
   contextlab.sandbox.worker` would pull in the package's `__init__.py`,
   which loads sentence-transformers / torch / numpy. Cold import alone
   exceeds the wall-clock timeout on a 2-second budget, and a strict
   `--as` (memory) limit crashes OpenBLAS before the worker even runs.
   The fix: invoke the worker by absolute path (`python
   <worker_path> --tool <name> --args-json <json>`). Only `worker.py`
   is loaded — stdlib only.
2. **`cpu_s` defaults to `null`, not `2`.** The brief lists `cpu_s: 2` as
   a *target*, but `prlimit --cpu=2` is in CPU-seconds and kills
   Python startup that includes OpenBLAS / numpy imports. With `cpu_s`
   null, `prlimit` is bypassed and the wall-clock `timeout_s` is the
   verified containment. Tests that want a CPU cap can pass a limits
   object with `cpu_s` set. Recorded here as a deliberate deviation
   from the yaml's example value.
3. **Memory limit (`memory_mb`) is recorded but not applied.** Same
   reason: Python startup imports blow past 256 MiB on OpenBLAS thread
   pools. The brief said the cap was best-effort ("may use it") so this
   is a no-op, not a violation. A `cpu_s`-only `prlimit` is still a
   useful containment signal and is used when configured.
4. **Test fixtures live in `worker.py`.** The brief says fixture tools
   are "only registered in tests." The cleanest implementation is to
   put the fixture handlers *inside* the worker process so the policy
   gate runs in the parent and the actual fixture work runs in the
   child. `read_secret` is allowed by the policy in tests; the path
   jail denies the read before the worker is launched.
5. **The executor unwraps one level of `{"result": ...}`.** Each handler
   returns its own payload (e.g. `python_calc` returns `{"result": 14}`,
   `read_secret` returns `{"path": ..., "content": ...}`). The worker
   wraps the handler's payload in `{"ok": True, "result": <payload>}`.
   The executor unwraps one level so the orchestrator sees the bare
   payload. `_last_calc_result` in the script policy reads
   `state.observations[*].content` (a JSON string) and pulls `result`
   directly — both paths stay consistent.
6. **Hard ban check.** No `shell` / `python -c` tool. No `eval()` on
   raw strings — `calc.evaluate` is an AST walk with a closed
   whitelist. No Docker, no bubblewrap, no required runtime — the
   subprocess backend works on a stock Python. `prlimit` is detected
   but optional.
7. **`trace/types.py` now requires `tool.exec` to carry `tool`, `backend`,
   `timeout_s`.** The brief's attribute set (`sandbox`, `exit_code`,
   `code`) is also set on every span. A test (`test_tool_exec_span_has_required_attrs`)
   reads back the in-memory exporter and asserts the contract.
8. **`UnknownToolError` propagation is preserved.** Slice 7's machine
   still catches `UnknownToolError` from any executor; the
   `SandboxedExecutor`'s policy raises `ToolError` (not
   `UnknownToolError`) for unregistered tools, and the in-process
   helper still raises `UnknownToolError`. The brief's test
   `test_unknown_tool_at_orchestrator_level_still_raises` pins this so
   Slice 9 can grade trajectory stop reasons uniformly.

### Not done here (later slices, not started)
- No trajectory eval suite (Slice 9 grades `artifacts/trajectories/*.json`).
- The LLM driver is still a skip-safe stub from Slice 7.

---

# ContextLab — Slice 9 Progress (Trajectory Eval Suite)

## Current Verified State
```
Date: 2026-09-15
Status: COMPLETE ✓
Default run:  python -m contextlab.evals run --suite trajectory --offline
              12/12 passed  pass_rate 1.000  step_cap_violations 0  unknown_tool_on_happy_path 0
              3/3 trajectory gates PASS
Mutation run: config/orchestrator.yaml max_steps: 6 -> 1
              1/12 passed  pass_rate 0.083  gate_trajectory_pass_rate FAIL  (exit 1)
Slices 1-8 unchanged: 259 pytest passed; --suite all --offline 13/13 gates PASS.
```

### Done Criteria (from brief §1)
- [x] `evals/trajectory_golden.jsonl` exists with t001 … t012 (12 cases, the
      §5 table).
- [x] `python -m contextlab.evals run --suite trajectory --offline` writes
      results into the shared report (`artifacts/eval_report.json`, same
      schema as Slice 3 — `suites.trajectory`, per-case `CaseResult` with
      `trace_id`).
- [x] `evals/gates.yaml` has a `trajectory` block that is enforced
      (`min_pass_rate: 0.85`, `max_step_cap_violations: 0`,
      `max_unknown_tool_on_happy_path: 0`); a failing gate exits 1.
- [x] `pytest tests/test_traj_evals.py -q` covers grader helpers and a
      fixture trajectory — 43 tests.
- [x] Mutation experiment is in this file (below), before/after recorded.
- [x] `--suite all --offline` includes trajectory and still runs
      retrieval/assembly/router/cache (6 suites, 13 gates).

### Mutation experiment (brief §6 / feat-j4)

**Default (shipped config `max_steps: 6`):**

| metric | value |
|--------|-------|
| trajectory cases | 12 |
| passed | 12 |
| pass_rate | **1.000** |
| step_cap_violations | 0 |
| unknown_tool_on_happy_path | 0 |
| `gate_trajectory_pass_rate` | PASS (1.000 >= 0.85) |
| `gate_trajectory_step_caps` | PASS (0 <= 0) |
| `gate_trajectory_unknown_tool_happy` | PASS (0 <= 0) |
| process exit | 0 (`=== EVAL PASSED ===`) |

**Mutation (`config/orchestrator.yaml` → `max_steps: 1`):**

| metric | value |
|--------|-------|
| trajectory cases | 12 |
| passed | **1** |
| pass_rate | **0.083** |
| step_cap_violations | 0 |
| unknown_tool_on_happy_path | 0 |
| `gate_trajectory_pass_rate` | **FAIL** (0.083 >= 0.85) |
| `gate_trajectory_step_caps` | PASS (0 <= 0) |
| `gate_trajectory_unknown_tool_happy` | PASS (0 <= 0) |
| process exit | 1 (`=== EVAL FAILED ===`) |

Which cases flip, and why: **every happy-path case fails** (t001–t008, t011,
t012) because the script policy's `last_step_action` fires a `finish` on
step 0 before any tool runs — `stop_reason` is still `done`, but
`required_tools` (retrieve / read_chunk / python_calc) never appeared and
the observation-substring checks fail. t009 stays PASS because it
*expects* `max_steps` — it is the cap case, so shrinking the cap is not a
regression for it. t010 fails because the unknown-tool fixture never gets
to emit its tool.

Reading of the mutation: the suite's signal is carried by the
**tool-presence and observation constraints**, not by `stop_reason` alone
— a run can stop "successfully" (`done`) while having done nothing, and
the graders catch that. That is the failure mode a trajectory suite exists
to catch; a final-answer-only eval would have scored the mutation as 12/12
passing (every `final_answer` is a string).

The mutation was applied by editing `config/orchestrator.yaml`, running the
suite, and restoring the file. `config/orchestrator.yaml` is back to
`max_steps: 6` and re-verified (12/12, 3/3 gates PASS). No golden was
tuned.

### What was built
- `evals/trajectory_golden.jsonl` — 12 cases (t001–t012) written as
  **constraints**, not step dumps: `expected_stop_reasons`,
  `max_steps_cap`, `required_tools`, `forbidden_tools`,
  `tool_order_strict`, `expected_citations`,
  `required/forbidden_observation_substrings`,
  `required/forbidden_final_answer_substrings`, `policy_name`,
  `settings_override`, `needs_llm`, `case_type`.
- `src/contextlab/evals/types.py` — `TrajectoryCase(EvalCase)` with the
  additive fields above. `EvalCase` untouched; Slices 3–6 goldens still
  parse.
- `src/contextlab/evals/graders_trajectory.py` — 12 deterministic graders
  over a `Trajectory`: `stop_reason`, `step_cap`, `required_tools`,
  `forbidden_tools`, `tool_order`, `citations`, `obs_contains`,
  `obs_forbids`, `final_contains`, `final_forbids`, `trace_present`,
  `prompt_tokens`. `grade()` runs all of them; a case passes when every
  active check passes.
- `src/contextlab/evals/suites_trajectory.py` — loads the golden file,
  runs each case through `orch.orchestrate` (real sandbox + tracer),
  attaches `trace_id`, aggregates the three gate metrics. Writes
  trajectories to a per-run tmp dir so `artifacts/trajectories/` stays a
  runtime path, not a checked-in artifact.
- `src/contextlab/orch/script_policy.py` — `FIXTURE_POLICIES` registry
  (`always_retrieve`, `unknown_tool`, `forbidden_calc`) plus
  `build_fixture_policy(name)`. A case's `policy_name` selects one; empty
  falls through to the production `ScriptedPolicy`. The brief allows this
  registry; there are no hidden ifs in the eval runner. A fourth candidate
  (`timeout`) was dropped: it would have called `sleep_forever`, which is
  not in `config/sandbox.yaml`'s tool list, so it would have produced
  `unknown_tool` rather than a timeout — a fixture that silently does the
  wrong thing is worse than no fixture. Slice 8's
  `test_timeout_kills_long_running` already exercises the real timeout path
  end-to-end (subprocess + `TimeoutExpired`).
- `src/contextlab/sandbox/executor.py` — one behavior change: the policy's
  `unknown_tool` denial now raises `UnknownToolError` (it previously
  returned a structured `ToolResult`). The orchestrator's Slice 7 catch
  then maps it to `stop_reason=unknown_tool`, which is what t010 grades.
  `code=unknown_tool` is still written to the `tool.exec` span.
- `src/contextlab/evals/runner.py` + `evals/__main__.py` — `trajectory`
  added to `--suite` choices, to the `all` list, and a `trajectory` branch
  in `apply_gates`.
- `evals/gates.yaml` — the `trajectory` block.
- `tests/test_traj_evals.py` — 43 tests: a grader class per check (pass +
  fail), `_referenced_chunk_ids` extraction, the fixture-policy registry,
  and an end-to-end suite smoke (12 cases, t005 uses `python_calc`, t009
  caps, t010 stop reason).
- `feature_list.json` — `feat-j1` … `feat-j5` (done).
- `AGENTS.md` — Slice 9 listed under "What", verify lines under
  "Run / Verify", four Slice 9 hard bans added.

### Real run output (default)
```
=== Eval Summary ===
  trajectory: 12/12 passed
    n_total: 12
    n_pass: 12
    n_fail: 0
    n_skip: 0
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

### Real per-case detail (default run, excerpts)
```
t001 stop_reason PASS stop_reason='done' expected=['done']
     required_tools PASS required=['retrieve'] seen=['retrieve','read_chunk']
     citations PASS expected=[error_codes::c0002, …] overlap=[incident_runbook::c0004, …]
t005 required_tools PASS required=['python_calc'] seen=['python_calc']
     forbidden_tools PASS forbidden=['retrieve'] found=[]
     obs_contains PASS required=['14'] missing=[]          output: 14
t009 stop_reason PASS stop_reason='max_steps' expected=['max_steps']
     step_cap PASS n_steps=1 cap=1 (orch_max_steps=1)
t010 stop_reason PASS stop_reason='unknown_tool' expected=['unknown_tool']
t011 obs_contains PASS required=['policy','disallowed'] missing=[]
```

### Spot checks (brief §7)
1. **t001 citations resolve to real chunk_ids.** Every chunk_id referenced
   by any trajectory in the run was checked against `data/chunks.jsonl`
   (34 ids): zero bogus ids. t001 references `incident_runbook::c0004`
   and `error_codes::c0002`, both real.
2. **t005 does not skip calc when the sandbox backend is subprocess.**
   `required_tools=['python_calc']` passes, `forbidden_tools=['retrieve']`
   passes (no fallback), `obs_contains=['14']` passes, `output: 14`. The
   `tool.exec` span shows `sandbox=true backend=subprocess`.
3. **Mutation report exists; goldens were not tuned.** The table above is
   the record; `config/orchestrator.yaml` is restored and re-verified.

### Findings and judgment calls (all written down)
1. **Grade what the trajectory *referenced*, not `trajectory.citations`.**
   For an orch run the assembler is called with `retrieve=False` and
   `tools=observations`, so `AssembledContext.citations` is empty — the
   retrieval-branch-only citation logic in Slice 2 (`assemble.py:211-216`)
   never runs. `graders_trajectory._referenced_chunk_ids` therefore reads
   both `read_chunk` action args and the `chunk_id`/`hits[*].chunk_id`
   fields out of observation payloads. That is the stable definition and
   it is documented on the function.
2. **`unknown_tool` had to be raised, not returned.** Slice 8's executor
   encoded every policy denial as a `ToolResult`, so a fixture calling
   `shell` produced six observations and `stop_reason=done`. The
   orchestrator only maps `UnknownToolError` to `stop_reason=unknown_tool`,
   so the `unknown_tool` code path was unreachable from the sandboxed
   executor. Fixed in `SandboxedExecutor.run` (raise for that one code);
   `test_sandbox.py::test_unknown_tool_propagates_to_orchestrator` and
   `…_span_records_code` pin it, and `test_unknown_tool_at_orchestrator_level_still_raises`
   keeps the in-process path pinned.
3. **`max_steps` precedence is explicit.** `suites_trajectory._build_request`
   resolves the cap as: case `settings_override.max_steps` (floored by) the
   global `CONTEXTLAB_ORCH_MAX_STEPS` env or `config/orchestrator.yaml`.
   Without that ordering, editing the config would have had no effect on
   the suite and the mutation would have been a no-op — which is exactly
   what the first mutation attempt showed (`12/12 passed` with
   `max_steps: 1`). Recorded because it is the kind of silent no-op that
   makes a "mutation experiment" meaningless.
4. **Constraint goldens, not step dumps.** A row asserts what must be
   true; it does not replay a recorded step list. Adding a hit to
   retrieve, or reordering two independent tool calls, does not break a
   case. The two fixture-shaped cases that could have been snapshot tests
   (t009 cap, t010 unknown tool) are also constraint rows.
5. **Trajectories written to a per-run tmp dir.** `run_suite` defaults
   `trajectory_dir` to `tempfile.mkdtemp(prefix="traj-evals-")` so a suite
   run does not accumulate files in `artifacts/trajectories/` (that path
   is gitignored and used by the CLI). Passing `trajectory_dir=` explicitly
   keeps the artifacts for inspection, which the tests do.
6. **`needs_llm` cases skip, they do not fail.** `offline=True` (the
   default and the only verified mode) records a `needs_llm` check that
   passes with `skipped`. No case in the shipped 12 sets it — the whole
   set is script-driver — but the field is wired so a future LLM-driver
   case does not break CI.
7. **The suite is 12 cases, no filler.** t001–t012 map one-to-one onto the
   brief's §5 table. t007 (identifier + calc in one query) was implemented
   as the brief's sanctioned alternative — a paraphrase of E-4471 — because
   the script policy has no combined branch; the case's `notes` says so.
   t011 likewise takes the brief's **policy** option over the timeout
   option: a fixture calls `python_calc` with an AST-rejected expression
   and the case asserts `obs_contains: ["policy", "disallowed"]`. The
   timeout path is not reproducible through the eval surface without
   registering `sleep_forever` in `config/sandbox.yaml`, which would put a
   test fixture in production config; it stays covered by Slice 8's
   `test_timeout_kills_long_running` instead.
8. **Hard ban check.** No step-dump goldens checked in. No LLM judge on
   tool order — every grader is a string/set comparison. No multi-agent
   case. The sandbox timeout tests were not touched (Slice 8's 35 tests
   still pass, one renamed assertion aside). Every case runs through
   `orch.orchestrate`, not a re-graded JSON file.

### The third trio is closed
retrieve → pack → eval → trace → route → cache → bounded loop → sandbox →
trajectory gates. Slice 9's mutation is recorded above.

### Not done here (later, not started)
- Prompt registry, streaming proxy, MCP, durable checkpoints — only if a
  named failure justifies one.
- The LLM driver is still a skip-safe stub from Slice 7; `needs_llm` cases
  are wired but unwritten.

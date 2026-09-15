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

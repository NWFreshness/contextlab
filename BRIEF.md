# Hermes brief — Slice 2: Token-budgeted Context Assembler

You are implementing Slice 2 of **ContextLab**. Slice 1 (hybrid retrieval) must already exist and its retrieval eval must have been run. If Slice 1 verify is failing, fix that first. Do not start Slice 3 eval-harness work except for keeping types compatible.

Read this whole file before writing code.

---

## 1. Goal and done-criteria

**Goal.** Build a context assembler that packs **instructions + retrieved hits + memory + tool results + the user query** into a token budget, drops lowest-priority content first, and emits an inspectable `AssembledContext` with citations that map to real chunk ids or tool ids.

**User.** A later generator (not required in this slice) and Slice 3 graders. Tyler should be able to see exactly which tokens the model would have received and why something was dropped.

**Constraints.**

- Same stack as Slice 1. Import `contextlab.retrieve`. Do not reimplement retrieval.
- No LangChain / LlamaIndex message-history helpers.
- Token counting must use the same tokenizer as chunking (`tiktoken` + encoding from config), not `len(text)//4` as the source of truth. A rough estimator may exist only as a fallback with a warning.
- Generation is optional and off by default. This slice is correct if it produces the packed prompt. An `--generate` flag may call an official SDK if `OPENAI_API_KEY` or equivalent is set; tests must pass without a key.
- Keep packing rules in `config/assembly.yaml` plus code. Prompts live in `prompts/`, not in chat history.

**Success metric.** Given a fixed budget B and a fixture of oversized inputs, the assembler:

- never emits `prompt_tokens > B`;
- always includes the user query and the system instructions (or fails closed with an error if those two alone exceed B);
- drops tool/memory/retrieval content by published priority, never silently truncating the middle of a citation;
- every citation in the packed prompt resolves to a `Hit.chunk_id` or a tool result id from the input.

**This slice is done only when all of these are true:**

- [ ] `python -m contextlab.assemble --query "..." --budget 800` prints a packed prompt and a drop report.
- [ ] `pytest tests/test_assemble.py -q` passes budget, priority, citation, and overflow tests.
- [ ] A fixture where retrieval+memory+tools exceed the budget produces a deterministic drop list recorded in `artifacts/assembly_examples.json`.
- [ ] Spot-checks in §8 were run and written into `progress.md`.
- [ ] README documents run + verify for this slice.

---

## 2. Design

**Recommendation.** One pure function:

```text
assemble(request: AssembleRequest) -> AssembledContext
```

No agent loop. No multi-hop. Retrieval is a single call you already have.

**Why.** The load-bearing skill is priority under a hard token ceiling. Frameworks hide drops. You need the drop report.

**Fallback.** If Slice 1 retrieve is slow because of embeddings, allow `--mode bm25` in the assemble CLI so tests stay fast. Default remains hybrid.

**Tradeoffs.**

- Reserve blocks (system, query, citation tax) rather than packing greedily from the front. Greedy-from-front silently kills memory or tools depending on concat order.
- Drop whole chunks, not mid-sentence, except for a last-resort memory summary hook that this slice may stub.
- Include a one-line citation after each retrieved block (`[policies::c0003]`). Citation tokens count against the retrieval budget.
- Do not summarize with an LLM in this slice. If memory will not fit, drop oldest items and list them in `dropped`.

**Priority (highest = dropped last):**

0. User query (must keep; if it does not fit with system, raise `BudgetError`)
1. System instructions
2. Tool results marked `must_keep=true`
3. Retrieved hits, in retrieved rank order
4. Memory items, newest first
5. Tool results with `must_keep=false`
6. Optional few-shot / extra instructions

When dropping retrieval hits, drop from the tail (lowest rank) first.

**Reserves in `config/assembly.yaml`:**

```yaml
budget_tokens: 2000
reserves:
  system: 300
  query: 200
  tools: 400
  memory: 300
  retrieval: 700   # remainder may be reallocated; see packing rule
citation_overhead_tokens: 16
```

Packing rule: start with configured reserves. If a bucket is under-full after packing, leftover tokens spill to retrieval, then memory. Never steal from system or query.

---

## 3. Shared contracts

Add to `src/contextlab/types.py` (do not fork Slice 1 types).

```python
class MemoryItem:
    memory_id: str
    text: str
    created_at: str        # ISO-8601
    priority: int = 0      # higher kept longer; default recency wins when tied

class ToolResult:
    tool_id: str
    name: str
    content: str
    must_keep: bool = False

class AssembleRequest:
    query: str
    system: str
    budget_tokens: int
    retrieve: bool = True
    retrieve_k: int = 8
    retrieve_mode: str = "hybrid"
    memory: list[MemoryItem] = []
    tools: list[ToolResult] = []
    extra_instructions: str | None = None

class PackedBlock:
    kind: str              # system|query|retrieval|memory|tool|extra
    ref_id: str            # chunk_id, memory_id, tool_id, or "system"/"query"
    text: str
    token_count: int
    kept: bool

class AssembledContext:
    query: str
    budget_tokens: int
    prompt_tokens: int
    prompt: str            # exact string that would be sent to the model
    blocks: list[PackedBlock]
    dropped: list[PackedBlock]
    citations: list[str]   # ref_ids that appear in the prompt
    retrieval: dict        # RetrieveResponse snapshot or empty
    settings: dict
```

Citation rule: every kept retrieval block must appear as its `chunk_id` in `citations` and in the prompt text. If a block is dropped, it must not be cited.

---

## 4. Files to add (do not rewrite Slice 1)

```
config/assembly.yaml
prompts/system_default.md
src/contextlab/tokenize.py      # single count_tokens(text) used everywhere
src/contextlab/assemble.py
src/contextlab/memory.py        # in-process list + optional JSONL store
tests/test_assemble.py
tests/fixtures/assemble_oversized.json
```

CLI: extend `src/contextlab/cli.py` or add `python -m contextlab.assemble`.

---

## 5. Implementation plan

### feat-a1 — Shared tokenizer

- `count_tokens(text: str) -> int` using the configured encoding.
- One test: a frozen string has a frozen count (record the number in the test).
- Verify: `pytest tests/test_assemble.py -k tokens -q`

### feat-a2 — Packer without retrieval

- Input: system, query, memory, tools, budget.
- Output: prompt + dropped.
- Tests:
  - query + system only, under budget → nothing dropped.
  - oversized memory → oldest dropped, query still present.
  - `must_keep` tool kept while a longer optional tool is dropped.
  - budget smaller than system+query → `BudgetError`.
- Verify: those tests pass.

### feat-a3 — Wire retrieve → pack

- Call Slice 1 `retrieve`. Convert `Hit`s to retrieval blocks. Apply retrieval reserve + spill.
- Prompt format (keep this exact section layout unless you document a change):

```
# Instructions
{system}

# Retrieved context
[{chunk_id} {doc_id}]
{text}

# Memory
[{memory_id}]
{text}

# Tool results
[{tool_id} {name}]
{content}

# User
{query}
```

Omit empty sections. Do not include dropped blocks.

- Verify: CLI on `what does E-4471 mean` with budget 800 produces at least one retrieval citation that exists in `data/chunks.jsonl`.

### feat-a4 — Drop report + example artifact

- Write `artifacts/assembly_examples.json` with 3 scripted cases:
  1. Comfortable budget (nothing dropped).
  2. Tight budget (retrieval tail dropped, citations still valid).
  3. Pathological budget (only system+query fit).
- Each case stores budget, prompt_tokens, kept refs, dropped refs.
- Verify: file exists and case 2 has `len(dropped) > 0` and `prompt_tokens <= budget`.

### feat-a5 — Optional generate flag

- Only if an API key is present. Send `AssembledContext.prompt` as a single user message or as system+user; document which.
- Log model, tokens in/out, latency to stdout. Do not pretend this is an eval.
- Tests must skip this path without a key.

---

## 6. Memory stub

Keep memory boring.

- `memory/session.jsonl` optional.
- CLI `--memory-file` loads items.
- If no file, tests pass in a list of `MemoryItem`.
- No embeddings, no memory retrieval. Assembler only packs what it is given. Slice 1 already owns retrieval.

Include 4–6 canned memory lines in a fixture that *conflict* with retrieved policy (e.g. memory says 14-day refund, corpus v3 says 30). The packer must still include both if they fit, and the drop report must show which one fell out when they do not. Do not “resolve” the conflict in this slice.

---

## 7. How to run

```bash
python -m contextlab.ingest          # if indexes missing
python -m contextlab.assemble \
  --query "How many days do I have to return a headset?" \
  --budget 800 \
  --memory-file tests/fixtures/memory.jsonl \
  --k 8
```

Print:

1. budget / prompt_tokens / dropped count
2. kept citations
3. the prompt
4. dropped ref_ids

---

## 8. How to verify

```bash
pytest tests/test_assemble.py -q
python -m contextlab.assemble --query "what does E-4471 mean" --budget 800
```

Spot-checks (record observations):

1. `--budget` equal to system+query tokens + 1 → `BudgetError` or only those two blocks; never a partial query.
2. Refund paraphrase + conflicting memory → both present when budget is large; drop report names the loser when budget is small.
3. Every `[chunk_id]` in the prompt exists in `data/chunks.jsonl`.
4. Changing `chunk_tokens` in retrieval config does not change `count_tokens` for the same packed string.

---

## 9. Hard bans

- Do not call an LLM to choose what to drop.
- Do not silently truncate a chunk in the middle to “make it fit” unless you add a `truncated: true` flag on that block and a test for it. Default is drop-whole-block.
- Do not invent citations.
- Do not add multi-hop retrieval, agents, or a router.
- Do not weaken Slice 1 tests.
- If retrieve returns nothing, say so in the prompt (`# Retrieved context\nNo passages retrieved.`) and continue.

---

## 10. State updates

Add feat-a1 … feat-a5 to `feature_list.json`. Point `AGENTS.md` verify at:

```bash
pytest -q && python -m contextlab.eval_retrieval
```

plus the assemble CLI example.

Update `progress.md` Current Verified State with budget-test evidence.

---

## 11. Skill this slice practices

Token budgeting, priority packing, citation integrity, inspectable context.

## 12. Next action after this slice

Stop. Hand off to `03-eval-harness.md`. The generator, if any, is a thin optional flag — Slice 3 owns grading.

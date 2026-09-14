# Hermes brief — Slice 1: Retrieval Stack

You are implementing Slice 1 of **ContextLab**, a from-scratch teaching system. Do not implement the context assembler or the full eval harness in this slice. Build a real hybrid retriever that later slices can import.

Read this whole file before writing code. Create the repo if it does not exist. After the session, leave `progress.md` and `feature_list.json` accurate.

---

## 1. Goal and done-criteria

**Goal.** Build a local retrieval stack: ingest markdown/text files → chunk with provenance → BM25 index + dense index → hybrid merge → optional rerank → ranked `Hit` list with scores and citations.

**User.** Tyler is learning AI engineering. The consumer is Slice 2 (context assembler) and Slice 3 (eval harness), plus a CLI he can run by hand.

**Constraints.**

- Python 3.11+, plain Python, pydantic models, pytest.
- Local corpus only. No hosted vector DB. No LangChain, LlamaIndex, Haystack, or equivalent.
- Allowed primitives: `tiktoken` or a documented tokenizer, `bm25s` or `rank-bm25`, `numpy`, a local embedding model via `sentence-transformers` **or** an official embeddings SDK behind an env var. Prefer a local embedding model so verify works offline.
- Secrets only in `.env`. Never commit keys.
- Pin versions in `pyproject.toml` or `requirements.txt`.
- Record chunking and retrieval settings in a checked-in `config/retrieval.yaml` (or `.json`). Do not bury them in code.

**Success metric.** On the checked-in golden set of ≥20 queries:

- Lexical queries (IDs, error codes, exact titles) get the supporting chunk in **hybrid top-5**.
- Paraphrase queries get the supporting chunk in **hybrid top-5**.
- A BM25-only vs dense-only vs hybrid comparison is printed. Hybrid must not be worse than the better of the two on overall recall@5. If it is, that is a bug or a config problem — fix it, do not hand-wave.

**This slice is done only when all of these are true:**

- [ ] `python -m contextlab.retrieve --query "..." --k 5` runs and prints hits with `doc_id`, `chunk_id`, scores, and a snippet.
- [ ] `python -m contextlab.ingest` (or equivalent) rebuilds indexes from `corpus/`.
- [ ] `pytest -q` passes unit tests for chunking, BM25, fusion, and provenance.
- [ ] `python -m contextlab.eval_retrieval` runs on `evals/retrieval_golden.jsonl`, writes `artifacts/retrieval_eval.json`, and prints recall@5 for bm25 / dense / hybrid.
- [ ] Failure cases below were spot-checked and noted in `progress.md`.
- [ ] README section for this slice says how to run and verify cold.

Do not claim done from a pretty print of one query.

---

## 2. Design (do this, not a larger design)

**Recommendation.** In-process indexes. Chunk once, write `data/chunks.jsonl`. Build a BM25 index over chunk text. Embed each chunk once, write `data/embeddings.npy` + `data/chunk_ids.json`. At query time: tokenize/embed the query, take top-N from each retriever, fuse with Reciprocal Rank Fusion (RRF), optionally rerank the fused top-20 with a score formula or a cross-encoder if it is already local.

**Why.** The skill is chunk policy, provenance, hybrid fusion, and measurement — not ANN infrastructure. For a few hundred chunks, brute-force cosine is correct and inspectable.

**Fallback.** If `sentence-transformers` is too heavy in this environment, use a hash-based bag-of-words dense stand-in **only for wiring tests**, and gate the real embedding path behind `EMBEDDING_MODEL`. Still implement the interface. Do not ship the stand-in as the default if a real model can be downloaded.

**Tradeoffs.**

- RRF over weighted score mixing: no score calibration required. Later you can add tuned weights.
- Fixed-size token chunks with overlap over semantic/LLM chunking: reproducible, cheap, good enough to see failure modes. Add a heading-aware splitter if it is <40 extra lines.
- No query rewrite in this slice. That hides retrieval bugs.
- No parent-document expansion yet. Return the chunk that was indexed. Assembler can request neighbors later via `chunk_id`.

**Data flow.**

```
corpus/*.md
  → ingest.chunk
  → chunks.jsonl
  → bm25 index + embeddings
  → retrieve(query, k, mode)
  → list[Hit]
```

---

## 3. Shared contracts (frozen for later slices)

Put these types in `src/contextlab/types.py`. Do not invent parallel shapes.

```python
# Conceptual schema — use pydantic. Field names must match.

class Chunk:
    chunk_id: str          # stable, e.g. "{doc_id}::c{n:04d}"
    doc_id: str            # filename stem
    doc_path: str
    text: str
    token_count: int
    start_char: int
    end_char: int
    section: str | None    # nearest heading if present
    metadata: dict

class Hit:
    chunk_id: str
    doc_id: str
    text: str
    score: float           # fused or reranked score used for final order
    scores: dict           # e.g. {"bm25": ..., "dense": ..., "rrf": ...}
    rank: int              # 1-based in the returned list
    citation: str          # "doc_id#chunk_id" or "doc_id § section"

class RetrieveRequest:
    query: str
    k: int = 5
    mode: str = "hybrid"   # "bm25" | "dense" | "hybrid"
    filters: dict | None   # optional metadata equality filters

class RetrieveResponse:
    query: str
    mode: str
    hits: list[Hit]
    settings: dict         # snapshot of config used
```

Golden row schema for this slice (`evals/retrieval_golden.jsonl`), one JSON object per line:

```json
{
  "id": "r001",
  "query": "...",
  "relevant_chunk_ids": ["policies::c0003"],
  "relevant_doc_ids": ["policies"],
  "intent": "lexical|paraphrase|table|version|negative",
  "notes": "why this case exists"
}
```

A hit is relevant if `chunk_id` is in `relevant_chunk_ids`, or if that list is empty and `doc_id` is in `relevant_doc_ids`. Prefer chunk-level labels.

---

## 4. Repo layout to create

```
contextlab/
  AGENTS.md
  README.md
  progress.md
  feature_list.json
  pyproject.toml          # or requirements.txt
  .env.example
  config/retrieval.yaml
  corpus/                 # 8–15 short docs, see §6
  evals/retrieval_golden.jsonl
  src/contextlab/
    __init__.py
    types.py
    config.py
    chunking.py
    ingest.py
    bm25_index.py
    dense_index.py
    fuse.py
    retrieve.py
    eval_retrieval.py
    cli.py
  tests/
    test_chunking.py
    test_bm25.py
    test_dense.py
    test_fuse.py
    test_retrieve.py
  artifacts/              # eval outputs, gitkeep only
```

Package name: `contextlab`. Import path: `from contextlab.retrieve import retrieve`.

---

## 5. Implementation plan (one feature at a time)

Work in this order. Do not start feature N+1 if N's verify command fails.

### feat-r1 — Project skeleton + types + config

- Create layout, packaging so `python -m contextlab.retrieve --help` works after install/editable install.
- Load `config/retrieval.yaml`:
  - `chunk_tokens` (default 256)
  - `chunk_overlap_tokens` (default 32)
  - `encoding` (e.g. `cl100k_base`)
  - `bm25_top_n` (default 50)
  - `dense_top_n` (default 50)
  - `rrf_k` (default 60)
  - `embedding_model`
  - `hybrid_k` (default 5)
- Verify: imports + config load test.

### feat-r2 — Chunker with provenance

- Split on headings first when a line matches `^#{1,6} `, then pack heading-bounded text into token windows with overlap.
- Every chunk has `start_char`, `end_char`, `section`, stable `chunk_id`.
- Re-ingesting the same file with the same config must produce the same `chunk_id`s.
- Unit tests: no empty chunks; overlap exists; a known fixture yields exact chunk_ids.
- Verify: `pytest tests/test_chunking.py -q`

### feat-r3 — Corpus + ingest

- Write the corpus in §6 exactly enough to support the golden queries. Do not generate random filler.
- `ingest` writes `data/chunks.jsonl` and prints chunk counts per doc.
- Verify: ingest runs; chunk count is stable across two runs.

### feat-r4 — BM25

- Index chunk text. Query returns ranked chunks with raw BM25 scores.
- Unit test: a query containing a unique token (`E-4471`) ranks that chunk first.
- Verify: `pytest tests/test_bm25.py -q`

### feat-r5 — Dense index

- Embed chunks. Persist vectors. Query by cosine similarity.
- Unit test: a paraphrase of a chunk ranks that chunk in top-5.
- Verify: `pytest tests/test_dense.py -q` (may download a model once; document that).

### feat-r6 — RRF hybrid + CLI

- Implement RRF: `score(d) = sum 1 / (rrf_k + rank_i(d))` over retrievers that returned `d`.
- CLI prints a table: rank, citation, rrf, bm25 rank, dense rank, 160-char snippet.
- Verify: one lexical query and one paraphrase query both return the labeled chunk in top-5.

### feat-r7 — Retrieval golden eval

- ≥20 cases covering intents in §6.
- Metric: recall@5 per mode, plus per-intent breakdown.
- Write `artifacts/retrieval_eval.json` with settings snapshot, timestamp, per-case hits, and aggregates.
- Verify: `python -m contextlab.eval_retrieval` and record the numbers in `progress.md`. If hybrid recall@5 < max(bm25, dense) overall, investigate before adding features.

Optional if time remains (do not swap this for feat-r7): a local cross-encoder rerank of the fused top-20. Keep it behind config `rerank: false`.

---

## 6. Corpus and golden-set design

Build a **tiny adversarial corpus**, not Wikipedia dumps. Target 8–15 docs, 2–8k tokens total.

Required documents (create these):

1. `refund_policy_v1.md` — old refund window (14 days). Mark as superseded in body.
2. `refund_policy_v3.md` — current refund window (30 days) and a unique policy ID `POL-REF-30`.
3. `error_codes.md` — table of codes including `E-4471` = "inventory reservation expired".
4. `onboarding_faq.md` — same facts as policies, different wording (paraphrase bait).
5. `shipping_sla.md` — numeric SLAs, including "West Coast ground = 5 business days".
6. `incident_runbook.md` — steps that mention `E-4471` and a wrong-looking neighbor code `E-4472`.
7. `boilerplate.md` — repeated marketing language that should *not* win retrieval.
8. `pricing.md` — a markdown table of plan names and limits.

Golden intents to cover (≥20 rows total):

- **lexical** — exact code, policy ID, plan name.
- **paraphrase** — "how long can I send something back" → v3 refund, not v1.
- **table** — a cell fact only present in a table.
- **version** — current policy vs superseded. Label v3 as relevant unless the query says "old" or "2022".
- **negative** — query about a topic not in corpus. `relevant_chunk_ids: []`. Retriever may still return hits; eval records that. Do not fake empty results.

Write `notes` on every golden row so a human can see why it exists.

---

## 7. How to run

Document the actual commands you implement. Defaults:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m contextlab.ingest
python -m contextlab.retrieve --query "what does E-4471 mean" --k 5 --mode hybrid
```

---

## 8. How to verify

```bash
pytest -q
python -m contextlab.eval_retrieval
```

Record in `progress.md`: command, exit code, recall@5 bm25/dense/hybrid, number of golden cases, model name, chunk settings.

Spot-check failure modes (write what you observed):

1. Query `E-4471` — BM25 should beat dense. Hybrid must still keep it top-5.
2. Paraphrase refund query — dense should beat BM25. Hybrid must still keep v3, not v1, in top-5.
3. Boilerplate-heavy query — marketing doc must not crowd out the policy chunk in hybrid top-5.
4. Re-ingest twice — chunk_ids identical.

---

## 9. Hard bans

- No LangChain / LlamaIndex / Haystack.
- No hosted vector DB.
- No query-rewrite LLM in this slice.
- No fake recall numbers. If eval cannot run, say so and leave status unfinished.
- Do not delete or shrink golden cases to make recall look good.
- Do not implement assembler, LLM generation, semantic cache, or agents.
- Do not rewrite the whole repo between features.

---

## 10. AGENTS.md / state you must write

Create a short root `AGENTS.md` for ContextLab:

- What: local hybrid retrieval + later assembler + evals.
- Run / verify commands above.
- One feature at a time from `feature_list.json`.
- Hard bans from §9.

`feature_list.json` starts with feat-r1 … feat-r7 as specified. Only mark a feature done when its verify command ran and evidence is a command + result.

`progress.md` must have a Current Verified State block and one session block per session.

---

## 11. Skill this slice practices

Chunking with provenance, lexical vs dense failure modes, RRF fusion, retrieval eval.

## 12. Next action after this slice

Stop. Hand off to `02-context-assembler.md`. Do not start packing prompts until hybrid recall@5 is measured and recorded.

"""Token-budgeted context assembler."""
import os
from pathlib import Path
from typing import Optional

import yaml

from contextlab.tokenize import count_tokens
from contextlab.trace import current_trace_id, prompt_attributes, start_span
from contextlab.types import (
    AssembleRequest,
    AssembledContext,
    MemoryItem,
    PackedBlock,
    ToolResult,
)


class BudgetError(Exception):
    """Raised when system+query alone exceed the budget."""
    pass


def load_assembly_config() -> dict:
    """Load assembly config from YAML."""
    config_path = Path(__file__).parent.parent.parent / "config" / "assembly.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_system_prompt(path: Optional[str] = None) -> str:
    """Load system prompt from file."""
    if path is None:
        path = Path(__file__).parent.parent.parent / "prompts" / "system_default.md"
    with open(path) as f:
        return f.read()


def _build_block(
    kind: str,
    ref_id: str,
    text: str,
    kept: bool = True,
) -> PackedBlock:
    """Build a PackedBlock with token count."""
    return PackedBlock(
        kind=kind,
        ref_id=ref_id,
        text=text,
        token_count=count_tokens(text),
        kept=kept,
    )


def assemble(request: AssembleRequest) -> AssembledContext:
    """
    Assemble a token-budgeted prompt from multiple sources.

    Priority (highest = dropped last):
    0. User query (must keep)
    1. System instructions
    2. Tool results marked must_keep=true
    3. Retrieved hits (rank order)
    4. Memory items (newest first)
    5. Tool results with must_keep=false
    6. Extra instructions

    Dropping: drop from tail (lowest priority) first.
    """
    config = load_assembly_config()
    reserves = config["reserves"]
    citation_overhead = config.get("citation_overhead_tokens", 16)

    # ── System block (reserve: system) ─────────────────────────────────────
    system_block = _build_block("system", "system", request.system)

    # ── Query block (reserve: query) ───────────────────────────────────────
    query_block = _build_block("query", "query", request.query)

    # Check if system + query alone exceed budget
    essential_tokens = system_block.token_count + query_block.token_count
    if essential_tokens > request.budget_tokens:
        raise BudgetError(
            f"system+query tokens ({essential_tokens}) exceed budget ({request.budget_tokens})"
        )

    # ── Retrieve (if enabled) ─────────────────────────────────────────────
    retrieval_data: dict = {}
    retrieval_blocks: list[PackedBlock] = []

    if request.retrieve:
        from contextlab.retrieve import retrieve, RetrieveRequest

        mode = request.retrieve_mode
        # Allow bm25 override for speed in tests
        if os.getenv("RETRIEVE_MODE"):
            mode = os.getenv("RETRIEVE_MODE")

        resp = retrieve(
            query=request.query,
            k=request.retrieve_k,
            mode=mode,
        )

        retrieval_data = {
            "query": resp.query,
            "mode": resp.mode,
            "hits": [h.model_dump() for h in resp.hits],
            "settings": resp.settings,
        }

        for hit in resp.hits:
            # Format: "[{chunk_id} {doc_id}]" header + text + citation overhead
            citation_header = f"[{hit.chunk_id} {hit.doc_id}]"
            block_text = f"{citation_header}\n{hit.text}"
            block = _build_block("retrieval", hit.chunk_id, block_text)
            retrieval_blocks.append(block)

    # ── Memory blocks (reserve: memory) ────────────────────────────────────
    memory_blocks: list[PackedBlock] = []
    for item in request.memory:
        block = _build_block("memory", item.memory_id, item.text)
        memory_blocks.append(block)

    # Sort: newest first (priority desc, then created_at desc)
    # The MemoryItem sort order handles this
    # memory_blocks are already in the order from request.memory (sorted externally)
    # But we need to sort: highest priority first; within same priority, newest first
    # Reverse of insertion order since we want newest-first
    memory_blocks_sorted = sorted(
        memory_blocks,
        key=lambda b: b.token_count,  # will re-sort below
    )

    # Actually sort by (priority desc, created_at desc) — request.memory should be pre-sorted
    # Just use the order given; caller is responsible for sorting

    # ── Tool blocks (reserve: tools) ──────────────────────────────────────
    tool_blocks_must_keep: list[PackedBlock] = []
    tool_blocks_optional: list[PackedBlock] = []

    for tool in request.tools:
        block = _build_block("tool", tool.tool_id, tool.content)
        if tool.must_keep:
            tool_blocks_must_keep.append(block)
        else:
            tool_blocks_optional.append(block)

    # Sort optional tools by token count ascending (drop longest first when needed)
    tool_blocks_optional.sort(key=lambda b: b.token_count)

    # ── Extra instructions block ───────────────────────────────────────────
    extra_block: Optional[PackedBlock] = None
    if request.extra_instructions:
        extra_block = _build_block("extra", "extra", request.extra_instructions)

    # ── Packing: build final blocks respecting priority ─────────────────────
    # The pack span is opened here and closed at the end of the function: the
    # pack hop has no child spans, and an exception mid-pack leaves the span
    # open for the trace finalizer to close as an error.
    pack_span = start_span(
        "assemble.pack",
        budget_tokens=request.budget_tokens,
        retrieve_enabled=request.retrieve,
        retrieved_ids=[b.ref_id for b in retrieval_blocks],
        n_memory=len(memory_blocks),
        n_tools=len(request.tools),
    )

    # Available budget after essentials
    available = request.budget_tokens - essential_tokens

    # Block bucket: (priority, blocks_list, reserve_key_or_None)
    # Priority 0=query, 1=system, 2=must_keep tools, 3=retrieval, 4=memory, 5=optional tools, 6=extra

    kept_blocks: list[PackedBlock] = []
    dropped_blocks: list[PackedBlock] = []
    citations: list[str] = []

    def try_add_block(block: PackedBlock) -> bool:
        """Try to add block to kept. Returns True if added."""
        if block.token_count <= available:
            kept_blocks.append(block)
            return True
        return False

    # 0. Query (already validated)
    kept_blocks.append(query_block)

    # 1. System
    if system_block.token_count <= available:
        kept_blocks.append(system_block)
    else:
        # Should not happen — already checked essentials
        raise BudgetError("system alone exceeds budget")

    # 2. Must-keep tools
    for block in tool_blocks_must_keep:
        if block.token_count <= available:
            kept_blocks.append(block)
            available -= block.token_count
        else:
            # Must-keep tools should not be dropped... but if truly impossible
            dropped_blocks.append(block)

    # 3. Retrieval blocks (sorted by rank — already in order)
    # Reserve for retrieval: at least some minimum per citation
    retrieval_reserve = reserves.get("retrieval", 700)

    for block in retrieval_blocks:
        # Include citation overhead
        total_cost = block.token_count + citation_overhead
        if total_cost <= available:
            kept_blocks.append(block)
            available -= total_cost
            citations.append(block.ref_id)
        else:
            dropped_blocks.append(block)

    # 4. Memory blocks (newest first — request.memory order)
    for block in memory_blocks:
        if block.token_count <= available:
            kept_blocks.append(block)
            available -= block.token_count
        else:
            dropped_blocks.append(block)

    # 5. Optional tools (shortest first — already sorted ascending)
    for block in tool_blocks_optional:
        if block.token_count <= available:
            kept_blocks.append(block)
            available -= block.token_count
        else:
            dropped_blocks.append(block)

    # 6. Extra instructions
    if extra_block and extra_block.token_count <= available:
        kept_blocks.append(extra_block)
        available -= extra_block.token_count

    # ── Build prompt string ────────────────────────────────────────────────
    # kept_blocks is in priority order (query, system, must_keep_tools, retrieval,
    # memory, optional_tools, extra). Prompt format wants: system, retrieval,
    # memory, tools, user. So we collect by kind then emit in right order.

    by_kind: dict[str, list[PackedBlock]] = {
        "system": [],
        "retrieval": [],
        "memory": [],
        "tool": [],
        "extra": [],
        "query": [],
    }
    for block in kept_blocks:
        by_kind[block.kind].append(block)

    prompt_parts: list[str] = []

    # # Instructions
    if by_kind["system"]:
        prompt_parts.append(f"# Instructions\n{by_kind['system'][0].text}\n")

    # # Retrieved context
    if by_kind["retrieval"]:
        for block in by_kind["retrieval"]:
            prompt_parts.append(f"\n# Retrieved context\n{block.text}\n")

    # # Memory
    if by_kind["memory"]:
        for block in by_kind["memory"]:
            prompt_parts.append(f"\n# Memory\n[{block.ref_id}]\n{block.text}\n")

    # # Tool results
    if by_kind["tool"]:
        for block in by_kind["tool"]:
            tool_name = next(
                (t.name for t in request.tools if t.tool_id == block.ref_id),
                block.ref_id,
            )
            prompt_parts.append(f"\n# Tool results\n[{block.ref_id} {tool_name}]\n{block.text}\n")

    # # Additional instructions
    if by_kind["extra"]:
        for block in by_kind["extra"]:
            prompt_parts.append(f"\n# Additional instructions\n{block.text}\n")

    # # User
    if by_kind["query"]:
        prompt_parts.append(f"\n# User\n{by_kind['query'][0].text}")

    prompt = "".join(prompt_parts).strip()

    # Handle empty retrieval case
    has_retrieval = len(by_kind["retrieval"]) > 0
    if request.retrieve and not has_retrieval and retrieval_blocks:
        prompt += "\n\n# Retrieved context\nNo passages retrieved.\n"

    prompt_tokens = count_tokens(prompt)

    # Close the pack hop with the ids needed to explain the prompt afterwards:
    # what stayed in context, what was dropped, and how big the prompt got.
    kept_ids = [b.ref_id for b in kept_blocks if b.kind in ("retrieval", "memory", "tool")]
    pack_span.set_attributes(
        prompt_tokens=prompt_tokens,
        kept_ids=kept_ids,
        dropped_ids=[b.ref_id for b in dropped_blocks],
        n_kept=len(kept_blocks),
        n_dropped=len(dropped_blocks),
        n_citations=len(citations),
        **prompt_attributes(prompt),
    )
    pack_span.finish()

    settings: dict = {"assembly_config": config}
    trace_id = current_trace_id()
    if trace_id:
        settings["trace_id"] = trace_id

    return AssembledContext(
        query=request.query,
        budget_tokens=request.budget_tokens,
        prompt_tokens=prompt_tokens,
        prompt=prompt,
        blocks=kept_blocks,
        dropped=dropped_blocks,
        citations=citations,
        retrieval=retrieval_data,
        settings=settings,
    )

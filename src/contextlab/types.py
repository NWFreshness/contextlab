"""Shared types for ContextLab retrieval stack."""
from pydantic import BaseModel, Field
from typing import Literal


class Chunk(BaseModel):
    """A chunk of text from a document with provenance tracking."""
    chunk_id: str = Field(description="stable chunk ID, e.g. '{doc_id}::c{n:04d}'")
    doc_id: str = Field(description="filename stem")
    doc_path: str = Field(description="path to source file")
    text: str = Field(description="chunk text content")
    token_count: int = Field(description="number of tokens in chunk")
    start_char: int = Field(description="character offset in source file")
    end_char: int = Field(description="end character offset in source file")
    section: str | None = Field(default=None, description="nearest heading if present")
    metadata: dict = Field(default_factory=dict)


class Hit(BaseModel):
    """A retrieval hit with scores and citation."""
    chunk_id: str
    doc_id: str
    text: str
    score: float = Field(description="fused or reranked score used for final order")
    scores: dict = Field(description="e.g. {'bm25': ..., 'dense': ..., 'rrf': ...}")
    rank: int = Field(description="1-based in the returned list")
    citation: str = Field(description="'doc_id#chunk_id' or 'doc_id § section'")


class RetrieveRequest(BaseModel):
    """Query request for retrieval."""
    query: str
    k: int = 5
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    filters: dict | None = None


class RetrieveResponse(BaseModel):
    """Response from retrieval with hits and settings."""
    query: str
    mode: str
    hits: list[Hit]
    settings: dict


# ─── Slice 2: Context Assembler ───────────────────────────────────────────────


class MemoryItem(BaseModel):
    """A single memory entry for the assembler."""
    memory_id: str
    text: str
    created_at: str  # ISO-8601
    priority: int = 0  # higher kept longer; default recency wins when tied


class ToolResult(BaseModel):
    """A tool call result for the assembler."""
    tool_id: str
    name: str
    content: str
    must_keep: bool = False


class AssembleRequest(BaseModel):
    """Request to assemble a context from multiple sources."""
    query: str
    system: str
    budget_tokens: int
    retrieve: bool = True
    retrieve_k: int = 8
    retrieve_mode: str = "hybrid"
    memory: list[MemoryItem] = []
    tools: list[ToolResult] = []
    extra_instructions: str | None = None


class PackedBlock(BaseModel):
    """A single block in the assembled prompt."""
    kind: str  # system|query|retrieval|memory|tool|extra
    ref_id: str  # chunk_id, memory_id, tool_id, or "system"/"query"
    text: str
    token_count: int
    kept: bool


class AssembledContext(BaseModel):
    """A fully assembled context ready for generation."""
    query: str
    budget_tokens: int
    prompt_tokens: int
    prompt: str  # exact string that would be sent to the model
    blocks: list[PackedBlock]
    dropped: list[PackedBlock]
    citations: list[str]  # ref_ids that appear in the prompt
    retrieval: dict  # RetrieveResponse snapshot or empty
    settings: dict

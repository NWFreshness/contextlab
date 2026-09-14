"""JSONL cache store: one entry per line, greppable, no server.

Embeddings live inline in the entry (rounded to 6 decimals). A sidecar .npy would
keep the file smaller, but inline vectors make an entry self-contained and
impossible to pair with the wrong row — worth a few KB at this scale.
"""
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from pydantic import BaseModel, Field

from contextlab.cache.fingerprint import CacheFingerprint, hash_text, normalize_query

EMBEDDING_PRECISION = 6


class CacheEntry(BaseModel):
    """One cached answer plus everything needed to decide it is still valid."""

    cache_id: str
    query: str
    embedding: Optional[list[float]] = None
    fingerprint: CacheFingerprint
    answer: str
    citations: list[str] = Field(default_factory=list)
    prompt_tokens: Optional[int] = None
    model: Optional[str] = None
    created_at: str
    trace_id: Optional[str] = None

    @property
    def normalized_query(self) -> str:
        """Exact-layer key for this entry."""
        return normalize_query(self.query)


def new_entry(
    *,
    query: str,
    fingerprint: CacheFingerprint,
    answer: str,
    citations: Sequence[str] = (),
    embedding: Optional[Sequence[float]] = None,
    model: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    trace_id: Optional[str] = None,
) -> CacheEntry:
    """Build an entry with a deterministic id (same request -> same cache_id)."""
    return CacheEntry(
        cache_id=hash_text(f"{fingerprint.key()}::{normalize_query(query)}"),
        query=query,
        embedding=(
            [round(float(value), EMBEDDING_PRECISION) for value in embedding]
            if embedding is not None
            else None
        ),
        fingerprint=fingerprint,
        answer=answer,
        citations=[str(citation) for citation in citations],
        prompt_tokens=prompt_tokens,
        model=model,
        created_at=datetime.now(timezone.utc).isoformat(),
        trace_id=trace_id,
    )


class CacheStore:
    """Append-only JSONL store. Unparseable lines are skipped, not fatal."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: Optional[list[CacheEntry]] = None

    def entries(self) -> list[CacheEntry]:
        """All entries (read once per process)."""
        if self._entries is None:
            self._entries = self._read()
        return self._entries

    def _read(self) -> list[CacheEntry]:
        if not self.path.exists():
            return []
        entries: list[CacheEntry] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(CacheEntry(**json.loads(line)))
            except (json.JSONDecodeError, ValueError):
                continue  # partial line from an interrupted write
        return entries

    def append(self, entry: CacheEntry) -> None:
        """Append one entry (creating the parent directory if needed)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")
        if self._entries is not None:
            self._entries.append(entry)

    def clear(self) -> None:
        """Forget everything (used by tests and `cache clear`)."""
        self._entries = []
        if self.path.exists():
            self.path.unlink()

    def stats(self) -> dict:
        """Counts by corpus version, retrieve mode and route."""
        entries = self.entries()
        return {
            "entries": len(entries),
            "with_embeddings": sum(1 for e in entries if e.embedding),
            "by_corpus_version": dict(Counter(e.fingerprint.corpus_version for e in entries)),
            "by_retrieve_mode": dict(Counter(e.fingerprint.retrieve_mode for e in entries)),
            "by_route": dict(Counter(str(e.fingerprint.route) for e in entries)),
            "distinct_queries": len({e.normalized_query for e in entries}),
        }

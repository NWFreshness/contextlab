"""Cache fingerprint — what has to match before a similar query may hit.

A cached answer is only valid for the same corpus, retrieve mode, token budget,
system prompt and (when the router ran) the same route and model. Near-duplicate
questions are not the only way to get a wrong answer: the same question with a
different budget or a different routed model is a different request.
"""
import hashlib
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.jsonl"
DEFAULT_SYSTEM_PATH = PROJECT_ROOT / "prompts" / "system_default.md"

_WHITESPACE = re.compile(r"\s+")
_TRAILING_PUNCT = " ?.!,\n\t"


class CacheFingerprint(BaseModel):
    """Everything that must be equal for an entry to be reusable."""

    corpus_version: str
    retrieve_mode: str
    budget_tokens: int
    system_hash: str
    route: Optional[str] = None
    model: Optional[str] = None

    def key(self) -> str:
        """Canonical comparable string (route/model only when they were known)."""
        parts = [
            f"corpus={self.corpus_version}",
            f"mode={self.retrieve_mode}",
            f"budget={self.budget_tokens}",
            f"system={self.system_hash}",
        ]
        if self.route is not None:
            parts.append(f"route={self.route}")
        if self.model is not None:
            parts.append(f"model={self.model}")
        return "|".join(parts)

    def differs_from(self, other: "CacheFingerprint") -> list[str]:
        """Field names that differ — the diagnostic behind `fingerprint_mismatch`."""
        return [
            field
            for field in ("corpus_version", "retrieve_mode", "budget_tokens", "system_hash", "route", "model")
            if getattr(self, field) != getattr(other, field)
        ]


def normalize_query(query: str) -> str:
    """Exact-layer key: casefold, collapse whitespace, drop trailing punctuation."""
    return _WHITESPACE.sub(" ", query).strip().casefold().rstrip(_TRAILING_PUNCT)


def hash_text(text: str) -> str:
    """16 hex chars of sha256 — enough to detect change, short enough to read."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def corpus_version(chunks_path: str | Path | None = None) -> str:
    """Hash of data/chunks.jsonl: changes on every ingest, so old entries die.

    Written into the fingerprint rather than remembered, so a cache can never
    serve an answer assembled from a corpus that no longer exists.
    """
    path = Path(chunks_path) if chunks_path is not None else DEFAULT_CHUNKS_PATH
    if not path.exists():
        return "no-corpus"
    return hash_text(path.read_bytes().decode("utf-8", errors="replace"))


def system_hash(system: str | None = None, system_path: str | Path | None = None) -> str:
    """Hash of the system prompt text (or of the file it came from)."""
    if system is None:
        path = Path(system_path) if system_path is not None else DEFAULT_SYSTEM_PATH
        system = path.read_text(encoding="utf-8") if path.exists() else ""
    return hash_text(system)


def fingerprint_for(
    *,
    retrieve_mode: str = "hybrid",
    budget_tokens: int = 800,
    system: str | None = None,
    route: str | None = None,
    model: str | None = None,
    corpus: str | None = None,
) -> CacheFingerprint:
    """Build the fingerprint for one request."""
    return CacheFingerprint(
        corpus_version=corpus if corpus is not None else corpus_version(),
        retrieve_mode=retrieve_mode,
        budget_tokens=int(budget_tokens),
        system_hash=system_hash(system),
        route=route,
        model=model,
    )


_FINGERPRINT_FIELDS = (
    "corpus_version",
    "retrieve_mode",
    "budget_tokens",
    "system_hash",
    "route",
    "model",
)


def fingerprint_from_dict(data: dict | None, *, system: str | None = None) -> CacheFingerprint:
    """Fingerprint from a partial dict (golden rows / CLI flags).

    Missing fields fall back to the live corpus + live system prompt; unknown keys
    are a hard error so a typo in a golden row cannot silently fingerprint nothing.
    """
    data = dict(data or {})
    unknown = [key for key in data if key not in _FINGERPRINT_FIELDS]
    if unknown:
        raise ValueError(
            f"unknown fingerprint field(s) {sorted(unknown)}; known: {list(_FINGERPRINT_FIELDS)}"
        )
    return CacheFingerprint(
        corpus_version=data.get("corpus_version") or corpus_version(),
        retrieve_mode=data.get("retrieve_mode") or "hybrid",
        budget_tokens=int(data.get("budget_tokens") or 800),
        system_hash=data.get("system_hash") or system_hash(system),
        route=data.get("route"),
        model=data.get("model"),
    )

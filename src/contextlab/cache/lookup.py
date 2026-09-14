"""Lookup: fingerprint check + exact layer + semantic layer + meaning guards.

Order matters and is deliberate:
  1. normalized-query exact match with a matching fingerprint  -> hit (exact)
  2. else nearest neighbour among same-fingerprint entries:
       guard fires                                  -> miss (guard:*)
       cosine >= threshold                          -> hit (semantic)
       otherwise                                    -> miss
  3. no usable same-fingerprint neighbour -> diagnose: fingerprint_mismatch or miss

A guard firing is always a miss. Similarity alone must never be enough.
"""
import re
from typing import Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field

from contextlab.cache.config import CacheConfig, load_cache_config
from contextlab.cache.fingerprint import CacheFingerprint, normalize_query
from contextlab.cache.store import CacheEntry, CacheStore, new_entry
from contextlab.router.signals import IDENTIFIER_RE  # one shared definition of "identifier"
from contextlab.trace import current_trace_id, start_span

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ENTITY_RE = re.compile(r"[A-Z][a-z]{2,}")
_PUNCT = "?,.:;!\"'()"

#: reasons a lookup can report
HIT_REASONS = ("exact", "semantic")
MISS_REASONS = ("miss", "fingerprint_mismatch")


class CacheLookup(BaseModel):
    """Outcome of one lookup. `score` on a miss is the similarity that was rejected."""

    hit: bool
    score: Optional[float] = None
    reason: str = "miss"
    entry: Optional[CacheEntry] = None
    detail: Optional[str] = None


class QueryEncoder:
    """Same stack as dense retrieval: sentence-transformers, cosine-normalized.

    One instance per model name per process; the cache does not reuse the dense
    retriever's instance because assemble() builds its own Retriever.
    """

    _shared: dict[str, "QueryEncoder"] = {}

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    @classmethod
    def shared(cls, model_name: str) -> "QueryEncoder":
        """Process-wide instance for a model name (avoids reloading per lookup)."""
        if model_name not in cls._shared:
            cls._shared[model_name] = cls(model_name)
        return cls._shared[model_name]

    def encode(self, text: str) -> list[float]:
        """Embed one string."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        vector = self._model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
        return [float(value) for value in vector]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, computed (not assumed) from the stored vectors."""
    vec_a = np.asarray(a, dtype=float)
    vec_b = np.asarray(b, dtype=float)
    denominator = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denominator)


def _words(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def identifiers_in(text: str) -> set[str]:
    """Identifiers (E-4471, POL-REF-30) in a query, lowercased."""
    found = set()
    for match in IDENTIFIER_RE.finditer(text):
        token = match.group(0)
        if any(char.isdigit() for char in token) and any(char.isalpha() for char in token):
            found.add(token.lower())
    return found


def capitalized_entities(text: str) -> set[str]:
    """Mid-sentence capitalized words — West/East Coast, Starter/Professional plan.

    The first word is skipped: sentence case would otherwise make every query look
    like it names an entity. All-caps acronyms (SLA) do not count either.
    """
    entities = set()
    for index, word in enumerate(text.split()):
        if index == 0:
            continue
        match = _ENTITY_RE.fullmatch(word.strip(_PUNCT))
        if match:
            entities.add(match.group(0).lower())
    return entities


def guard_reason(probe: str, seed: str, config: CacheConfig) -> Optional[str]:
    """Why these two questions must not share an answer, or None.

    Deterministic token logic — never an LLM judge. A guard firing is a miss, and
    every guard only fires on tokens the *probe* introduces, so a guard can cost a
    hit but can never cause a wrong hit.
    """
    guards = config.guards or {}
    probe_words, seed_words = _words(probe), _words(seed)

    if guards.get("identifiers_must_match", True):
        missing = identifiers_in(probe) - identifiers_in(seed)
        if missing:
            return f"guard:identifier_mismatch {' '.join(sorted(missing))}"

    if guards.get("numbers_must_match", True):
        missing = {w for w in probe_words if w.isdigit()} - {w for w in seed_words if w.isdigit()}
        if missing:
            return f"guard:number_mismatch {' '.join(sorted(missing))}"

    if guards.get("capitalized_entities_must_match", True):
        missing = capitalized_entities(probe) - capitalized_entities(seed)
        if missing:
            return f"guard:entity_mismatch {' '.join(sorted(missing))}"

    groups = guards.get("version_token_groups") or []
    if groups:
        probe_groups = {i for i, group in enumerate(groups) if probe_words & {w.lower() for w in group}}
        seed_groups = {i for i, group in enumerate(groups) if seed_words & {w.lower() for w in group}}
        if probe_groups and seed_groups and probe_groups != seed_groups:
            return "guard:version_conflict"

    return None


class Cache:
    """Lookup/write façade. Every call is traced (`cache.lookup`, `cache.write`)."""

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        store: Optional[CacheStore] = None,
        encoder: Optional[QueryEncoder] = None,
    ) -> None:
        self.config = config or load_cache_config()
        self.store = store if store is not None else CacheStore(self.config.resolved_path())
        self._encoder = encoder

    @property
    def encoder(self) -> QueryEncoder:
        """Lazy: the embedding model is not loaded until a lookup needs it."""
        if self._encoder is None:
            self._encoder = QueryEncoder.shared(self.config.resolved_embedding_model())
        return self._encoder

    def embed(self, text: str) -> list[float]:
        """Embed query text (queries only — never the packed prompt)."""
        return self.encoder.encode(text)

    def _fingerprint_problem(self, entry: CacheEntry, fingerprint: CacheFingerprint) -> Optional[str]:
        """None when the entry is reusable for this request, else why not."""
        if not self.config.require_fingerprint:
            return None
        differing = entry.fingerprint.differs_from(fingerprint)
        return f"fingerprint differs on {', '.join(differing)}" if differing else None

    def _entry_vector(self, entry: CacheEntry) -> list[float]:
        """Stored vector, or recompute from the stored query (JSONL may omit it)."""
        return entry.embedding if entry.embedding is not None else self.embed(entry.query)

    def lookup(self, query: str, fingerprint: CacheFingerprint) -> CacheLookup:
        """Look up one query under one fingerprint, with a `cache.lookup` span."""
        with start_span(
            "cache.lookup", threshold=self.config.threshold, require_fingerprint=self.config.require_fingerprint
        ) as span:
            result = self._lookup(query, fingerprint)
            span.set_attributes(
                hit=result.hit,
                score=result.score,
                reason=result.reason,
                cache_id=result.entry.cache_id if result.entry is not None else None,
                detail=result.detail,
                fingerprint=fingerprint.key(),
                entries_scanned=len(self.store.entries()),
            )
            return result

    def _lookup(self, query: str, fingerprint: CacheFingerprint) -> CacheLookup:
        entries = self.store.entries()
        if not entries:
            return CacheLookup(hit=False, score=None, reason="miss", detail="cache is empty")

        probe_norm = normalize_query(query)
        exact_mismatch: Optional[str] = None

        # ── layer 1: exact normalized query (no embedding cost) ────────────────
        if self.config.exact_layer:
            for entry in entries:
                if entry.normalized_query != probe_norm:
                    continue
                problem = self._fingerprint_problem(entry, fingerprint)
                if problem is None:
                    return CacheLookup(
                        hit=True, score=None, reason="exact", entry=entry,
                        detail="normalized query match",
                    )
                exact_mismatch = f"identical text but {problem}"

        # ── layer 2: nearest neighbour with a matching fingerprint ────────────
        probe_vector = self.embed(query)
        best_same: Optional[tuple[float, CacheEntry]] = None
        best_any: Optional[tuple[float, CacheEntry]] = None

        for entry in entries:
            score = cosine(probe_vector, self._entry_vector(entry))
            if best_any is None or score > best_any[0]:
                best_any = (score, entry)
            if self._fingerprint_problem(entry, fingerprint) is None:
                if best_same is None or score > best_same[0]:
                    best_same = (score, entry)

        if best_same is not None:
            score, entry = best_same
            guard = guard_reason(query, entry.query, self.config)
            if guard:
                return CacheLookup(
                    hit=False, score=score, reason=guard, entry=None,
                    detail=f"similar to {entry.cache_id} ({entry.query!r}) but the guard fired",
                )
            if score >= self.config.threshold:
                return CacheLookup(
                    hit=True, score=score, reason="semantic", entry=entry,
                    detail=f"cosine {score:.4f} >= {self.config.threshold}",
                )
            return CacheLookup(
                hit=False, score=score, reason="miss", entry=None,
                detail=f"best cosine {score:.4f} < {self.config.threshold}"
                       + (f"; {exact_mismatch}" if exact_mismatch else ""),
            )

        # ── nothing reusable: say which wall we hit ───────────────────────────
        if exact_mismatch:
            return CacheLookup(hit=False, score=None, reason="fingerprint_mismatch", detail=exact_mismatch)

        if best_any is not None:
            score, entry = best_any
            problem = self._fingerprint_problem(entry, fingerprint)
            if problem and score >= self.config.threshold:
                return CacheLookup(
                    hit=False, score=score, reason="fingerprint_mismatch", entry=None,
                    detail=f"cosine {score:.4f} but {problem} (crossed {entry.cache_id})",
                )

        return CacheLookup(hit=False, score=None, reason="miss", detail="no fingerprint-compatible entry")

    def write(
        self,
        query: str,
        fingerprint: CacheFingerprint,
        answer: str,
        citations: Sequence[str] = (),
        model: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        embedding: Optional[Sequence[float]] = None,
    ) -> CacheEntry:
        """Store one answer under the fingerprint it was produced for."""
        with start_span("cache.write", query_chars=len(query)) as span:
            vector = embedding if embedding is not None else self.embed(query)
            entry = new_entry(
                query=query,
                fingerprint=fingerprint,
                answer=answer,
                citations=citations,
                embedding=vector,
                model=model,
                prompt_tokens=prompt_tokens,
                trace_id=current_trace_id(),
            )
            self.store.append(entry)
            span.set_attributes(
                cache_id=entry.cache_id,
                answer_chars=len(answer),
                citations=len(entry.citations),
                embedding_dim=len(entry.embedding or []),
                fingerprint=fingerprint.key(),
                entries=len(self.store.entries()),
            )
            return entry

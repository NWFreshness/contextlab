"""Slice 6 — semantic cache: a hit needs a similar query *and* the same settings.

    from contextlab.cache import Cache, CacheLookup, fingerprint_for

    cache = Cache()
    lookup = cache.lookup(query, fingerprint_for(retrieve_mode="hybrid", budget_tokens=800))

Similarity alone is never enough. A hit also requires an identical fingerprint
(corpus version, retrieve mode, budget, system prompt, route/model) and must clear
the meaning guards (identifiers, numbers, version words). Local JSONL store, no
server. The assemble/generate wiring lives in `contextlab.cache.answer`, which is
deliberately *not* imported here so importing the cache core stays cheap.
"""
from contextlab.cache.config import CacheConfig, CacheConfigError, load_cache_config
from contextlab.cache.fingerprint import (
    CacheFingerprint,
    corpus_version,
    fingerprint_for,
    fingerprint_from_dict,
    hash_text,
    normalize_query,
    system_hash,
)
from contextlab.cache.lookup import (
    HIT_REASONS,
    MISS_REASONS,
    Cache,
    CacheLookup,
    QueryEncoder,
    cosine,
    guard_reason,
    identifiers_in,
)
from contextlab.cache.store import CacheEntry, CacheStore, new_entry

__all__ = [
    "HIT_REASONS",
    "MISS_REASONS",
    "Cache",
    "CacheConfig",
    "CacheConfigError",
    "CacheEntry",
    "CacheFingerprint",
    "CacheLookup",
    "CacheStore",
    "QueryEncoder",
    "corpus_version",
    "cosine",
    "fingerprint_for",
    "fingerprint_from_dict",
    "guard_reason",
    "hash_text",
    "identifiers_in",
    "load_cache_config",
    "new_entry",
    "normalize_query",
    "system_hash",
]

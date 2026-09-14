"""Shared pytest fixtures.

Every test run is kept out of the repo's data/ directory:
  - CONTEXTLAB_TRACE_DIR redirects the tracer's JSONL store
  - CONTEXTLAB_CACHE_PATH redirects the cache store

Without the second one, a real seeded data/cache.jsonl changes CLI test outcomes
(a probe that used to reach generate starts hitting the cache), which is exactly
the kind of cross-test coupling this fixture exists to prevent.
"""
import pytest

from contextlab.trace import reset as reset_tracer


@pytest.fixture(autouse=True)
def isolated_trace_dir(tmp_path, monkeypatch):
    """Redirect the default JSONL trace directory to a per-test tmp dir."""
    monkeypatch.setenv("CONTEXTLAB_TRACE_DIR", str(tmp_path / "traces"))
    reset_tracer()
    yield tmp_path / "traces"
    reset_tracer()


@pytest.fixture(autouse=True)
def isolated_cache_store(tmp_path, monkeypatch):
    """Redirect the cache store to a per-test tmp file (never data/cache.jsonl)."""
    monkeypatch.setenv("CONTEXTLAB_CACHE_PATH", str(tmp_path / "cache" / "cache.jsonl"))
    yield tmp_path / "cache" / "cache.jsonl"

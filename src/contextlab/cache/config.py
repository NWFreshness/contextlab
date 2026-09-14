"""Cache configuration: config/cache.yaml, validated, with safe defaults."""
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_CONFIG = PROJECT_ROOT / "config" / "cache.yaml"


class CacheConfigError(Exception):
    """Raised when the cache config cannot be honoured."""


class CacheConfig(BaseModel):
    """Parsed config/cache.yaml."""

    enabled: bool = True
    path: str = "data/cache.jsonl"
    threshold: float = 0.92
    embedding_model: Optional[str] = None
    require_fingerprint: bool = True
    exact_layer: bool = True
    guards: dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = None

    def resolved_path(self) -> Path:
        """Absolute cache path (relative paths resolve against the repo root)."""
        path = Path(self.path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def resolved_embedding_model(self) -> str:
        """cache.yaml override, else the dense retrieval model (same stack)."""
        if self.embedding_model:
            return self.embedding_model
        from contextlab.config import get_config as get_retrieval_config

        return get_retrieval_config()["embedding_model"]


def load_cache_config(path: str | Path | None = None) -> CacheConfig:
    """Load and validate config/cache.yaml. Malformed config fails loud.

    Environment overrides (used by tests to stay off the real store):
      CONTEXTLAB_CACHE_PATH     -> store path
      CONTEXTLAB_CACHE_ENABLED  -> 0/1
    """
    config_path = Path(path) if path is not None else DEFAULT_CACHE_CONFIG
    if not config_path.exists():
        raise CacheConfigError(f"cache config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    threshold = raw.get("threshold", 0.92)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise CacheConfigError(f"{config_path}: threshold must be a number, got {threshold!r}") from exc
    if not 0.0 <= threshold <= 1.0:
        raise CacheConfigError(f"{config_path}: threshold must be between 0 and 1, got {threshold}")

    guards = raw.get("guards") or {}
    groups = guards.get("version_token_groups") or []
    if not isinstance(groups, list) or any(not isinstance(g, list) for g in groups):
        raise CacheConfigError(
            f"{config_path}: guards.version_token_groups must be a list of word lists"
        )

    enabled = bool(raw.get("enabled", True))
    env_enabled = os.environ.get("CONTEXTLAB_CACHE_ENABLED", "").strip().lower()
    if env_enabled in ("1", "true", "yes", "on"):
        enabled = True
    elif env_enabled in ("0", "false", "no", "off"):
        enabled = False

    store_path = str(raw.get("path") or "data/cache.jsonl")
    if os.environ.get("CONTEXTLAB_CACHE_PATH"):
        store_path = os.environ["CONTEXTLAB_CACHE_PATH"]

    return CacheConfig(
        enabled=enabled,
        path=store_path,
        threshold=threshold,
        embedding_model=raw.get("embedding_model"),
        require_fingerprint=bool(raw.get("require_fingerprint", True)),
        exact_layer=bool(raw.get("exact_layer", True)),
        guards=dict(guards),
        source=str(config_path),
    )

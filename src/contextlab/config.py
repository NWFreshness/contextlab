"""Configuration loader for ContextLab."""
import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load retrieval config from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "retrieval.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Override from environment
    if os.getenv("EMBEDDING_MODEL"):
        config["embedding_model"] = os.getenv("EMBEDDING_MODEL")
    if os.getenv("BM25_TOP_N"):
        config["bm25_top_n"] = int(os.getenv("BM25_TOP_N"))
    if os.getenv("DENSE_TOP_N"):
        config["dense_top_n"] = int(os.getenv("DENSE_TOP_N"))
    if os.getenv("RRF_K"):
        config["rrf_k"] = int(os.getenv("RRF_K"))
    if os.getenv("HYBRID_K"):
        config["hybrid_k"] = int(os.getenv("HYBRID_K"))
    if os.getenv("RERANK"):
        config["rerank"] = os.getenv("RERANK").lower() in ("true", "1", "yes")
    
    return config


# Global config instance
_config: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    """Get cached config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config

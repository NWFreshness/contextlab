"""Token counting using tiktoken — single source of truth for all token counts."""
from pathlib import Path

import tiktoken

from contextlab.config import get_config


_encoder = None


def get_encoder():
    """Lazily create a tiktoken encoder from config."""
    global _encoder
    if _encoder is None:
        config = get_config()
        _encoder = tiktoken.get_encoding(config.get("encoding", "cl100k_base"))
    return _encoder


def count_tokens(text: str) -> int:
    """Count tokens in text using the configured tiktoken encoding."""
    encoder = get_encoder()
    return len(encoder.encode(text))


def count_tokens_raw(text: str, encoding: str = "cl100k_base") -> int:
    """Count tokens with an explicit encoding (for tests/fixtures)."""
    encoder = tiktoken.get_encoding(encoding)
    return len(encoder.encode(text))

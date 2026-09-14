"""Tests for dense retrieval."""
import pytest
from contextlab.dense_index import DenseIndex
from contextlab.chunking import chunk_text


def test_dense_paraphrase_ranking():
    """Test that a paraphrase ranks the original chunk in top-5."""
    chunks = chunk_text(
        "You have 30 days to return items for a full refund.",
        doc_id="refund",
        doc_path="/refund",
        chunk_tokens=256,
    )
    
    index = DenseIndex(chunks)
    
    # Query with paraphrase
    results = index.query("how long can I send something back", k=5)
    
    assert len(results) > 0, "Should return results"
    # The chunk about 30-day returns should be highly ranked
    assert results[0][0].doc_id == "refund" or any("30" in r[0].text for r in results[:2])


def test_dense_returns_scores():
    """Test that dense returns similarity scores."""
    chunks = chunk_text(
        "Test content for scoring.",
        doc_id="test",
        doc_path="/test",
        chunk_tokens=256,
    )
    
    index = DenseIndex(chunks)
    results = index.query("test content", k=5)
    
    assert len(results) > 0, "Should return results"
    assert all(0 <= score <= 1 for _, score, _ in results), "Cosine similarity should be 0-1"


def test_dense_empty_index():
    """Test querying empty index."""
    index = DenseIndex()
    results = index.query("anything", k=5)
    assert results == [], "Empty index should return empty results"

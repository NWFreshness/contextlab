"""Tests for BM25 retrieval."""
import pytest
from contextlab.bm25_index import BM25Index
from contextlab.chunking import chunk_text


def test_bm25_ranks_unique_token_first():
    """Test that a unique token ranks its chunk first."""
    # Create chunks with a unique term
    chunks = chunk_text(
        "E-4471 is the code for inventory reservation expired. This is a common error.",
        doc_id="test",
        doc_path="/test",
        chunk_tokens=256,
    )
    
    chunks.extend(chunk_text(
        "Other error codes include E-1001 and E-1002 for payment issues.",
        doc_id="test2",
        doc_path="/test2",
        chunk_tokens=256,
    ))
    
    index = BM25Index(chunks)
    
    # Query for the unique token
    results = index.query("E-4471", k=5)
    
    assert len(results) > 0, "Should return results"
    assert "E-4471" in results[0][0].text, "Chunk with E-4471 should rank first"


def test_bm25_returns_scores():
    """Test that BM25 returns scores."""
    chunks = chunk_text(
        "Test document with some content here.",
        doc_id="test",
        doc_path="/test",
        chunk_tokens=256,
    )
    
    index = BM25Index(chunks)
    results = index.query("test", k=5)
    
    assert all(score > 0 for _, score, _ in results), "BM25 scores should be positive"


def test_bm25_empty_index():
    """Test querying empty index."""
    index = BM25Index()
    results = index.query("anything", k=5)
    assert results == [], "Empty index should return empty results"

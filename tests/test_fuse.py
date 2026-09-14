"""Tests for RRF fusion."""
import pytest
from contextlab.fuse import reciprocal_rank_fusion
from contextlab.chunking import chunk_text


def test_rrf_combines_results():
    """Test that RRF combines BM25 and dense results."""
    chunk1 = chunk_text("E-4471 means inventory reservation expired.", doc_id="errors", doc_path="/errors")[0]
    chunk2 = chunk_text("Other error codes are E-1001, E-1002.", doc_id="errors2", doc_path="/errors2")[0]
    chunk3 = chunk_text("The weather is nice today.", doc_id="weather", doc_path="/weather")[0]
    
    bm25_results = [
        (chunk1, 10.0, 1),  # E-4471 chunk ranks first in BM25
        (chunk2, 5.0, 2),
    ]
    
    dense_results = [
        (chunk1, 0.9, 1),   # Same chunk ranks first in dense
        (chunk3, 0.85, 2),  # Weather chunk ranks second in dense
    ]
    
    hits = reciprocal_rank_fusion(bm25_results, dense_results, rrf_k=60, k=5)
    
    assert len(hits) >= 1, "Should return at least one hit"
    assert hits[0].chunk_id == chunk1.chunk_id, "E-4471 chunk should rank first"


def test_rrf_preserves_ranking():
    """Test that RRF preserves ranking when both retrievers agree."""
    chunk1 = chunk_text("exact match document", doc_id="d1", doc_path="/d1")[0]
    chunk2 = chunk_text("partial match document", doc_id="d2", doc_path="/d2")[0]
    
    bm25_results = [(chunk1, 10.0, 1), (chunk2, 5.0, 2)]
    dense_results = [(chunk1, 0.9, 1), (chunk2, 0.8, 2)]
    
    hits = reciprocal_rank_fusion(bm25_results, dense_results, rrf_k=60, k=2)
    
    assert hits[0].chunk_id == chunk1.chunk_id
    assert hits[1].chunk_id == chunk2.chunk_id


def test_rrf_includes_scores():
    """Test that hits include individual retriever scores."""
    chunk1 = chunk_text("test document", doc_id="d1", doc_path="/d1")[0]
    
    bm25_results = [(chunk1, 10.0, 1)]
    dense_results = [(chunk1, 0.9, 1)]
    
    hits = reciprocal_rank_fusion(bm25_results, dense_results, rrf_k=60, k=1)
    
    assert "bm25" in hits[0].scores
    assert "dense" in hits[0].scores
    assert "rrf" in hits[0].scores
    assert hits[0].scores["bm25"] == 10.0
    assert hits[0].scores["dense"] == 0.9

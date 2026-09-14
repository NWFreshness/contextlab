"""Tests for full retrieval pipeline."""
import pytest
from contextlab.retrieve import Retriever
from contextlab.types import RetrieveRequest


def test_retrieve_bm25_mode():
    """Test BM25-only retrieval mode."""
    retriever = Retriever()
    request = RetrieveRequest(query="E-4471 error", k=5, mode="bm25")
    response = retriever.retrieve(request)
    
    assert response.mode == "bm25"
    assert len(response.hits) <= 5
    assert all(h.scores["bm25"] > 0 for h in response.hits)


def test_retrieve_dense_mode():
    """Test dense-only retrieval mode."""
    retriever = Retriever()
    request = RetrieveRequest(query="return policy", k=5, mode="dense")
    response = retriever.retrieve(request)
    
    assert response.mode == "dense"
    assert len(response.hits) <= 5


def test_retrieve_hybrid_mode():
    """Test hybrid retrieval mode."""
    retriever = Retriever()
    request = RetrieveRequest(query="E-4471", k=5, mode="hybrid")
    response = retriever.retrieve(request)
    
    assert response.mode == "hybrid"
    assert len(response.hits) <= 5
    # Hybrid should have RRF scores
    assert all("rrf" in h.scores for h in response.hits)


def test_retrieve_returns_provenance():
    """Test that hits include provenance information."""
    retriever = Retriever()
    request = RetrieveRequest(query="refund policy", k=5, mode="hybrid")
    response = retriever.retrieve(request)
    
    for hit in response.hits:
        assert hit.chunk_id
        assert hit.doc_id
        assert hit.citation
        assert hit.text

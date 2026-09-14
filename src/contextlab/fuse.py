"""Reciprocal Rank Fusion (RRF) for combining retrieval results."""
from typing import Optional

from contextlab.types import Chunk, Hit


def reciprocal_rank_fusion(
    bm25_results: list[tuple[Chunk, float, int]],
    dense_results: list[tuple[Chunk, float, int]],
    rrf_k: int = 60,
    k: int = 5,
) -> list[Hit]:
    """
    Fuse BM25 and dense results using Reciprocal Rank Fusion.
    
    Score formula: score(d) = sum 1 / (rrf_k + rank_i(d)) for each retriever that returned d.
    """
    # Build rank dictionaries
    rrf_scores: dict[str, dict] = {}
    
    for chunk, raw_score, rank in bm25_results:
        chunk_id = chunk.chunk_id
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = {
                "chunk": chunk,
                "rrf": 0.0,
                "bm25_score": raw_score,
                "bm25_rank": rank,
                "dense_rank": None,
            }
        rrf_scores[chunk_id]["rrf"] += 1.0 / (rrf_k + rank)
    
    for chunk, raw_score, rank in dense_results:
        chunk_id = chunk.chunk_id
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = {
                "chunk": chunk,
                "rrf": 0.0,
                "bm25_rank": None,
                "dense_score": raw_score,
                "dense_rank": rank,
            }
        else:
            rrf_scores[chunk_id]["dense_score"] = raw_score
            rrf_scores[chunk_id]["dense_rank"] = rank
        rrf_scores[chunk_id]["rrf"] += 1.0 / (rrf_k + rank)
    
    # Sort by RRF score descending
    sorted_chunks = sorted(rrf_scores.values(), key=lambda x: x["rrf"], reverse=True)
    
    # Build Hit list
    hits = []
    for rank, data in enumerate(sorted_chunks[:k], 1):
        chunk = data["chunk"]
        citation = f"{chunk.doc_id} § {chunk.section}" if chunk.section else chunk.doc_id
        
        hit = Hit(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            text=chunk.text[:200],  # Snippet
            score=data["rrf"],
            scores={
                "rrf": data["rrf"],
                "bm25": data.get("bm25_score", 0.0),
                "dense": data.get("dense_score", 0.0),
            },
            rank=rank,
            citation=citation,
        )
        hits.append(hit)
    
    return hits

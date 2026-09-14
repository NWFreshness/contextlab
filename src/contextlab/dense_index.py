"""Dense index using sentence-transformers for semantic retrieval."""
import json
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from contextlab.types import Chunk


class DenseIndex:
    """Dense retriever using sentence-transformers embeddings."""
    
    def __init__(self, chunks: Optional[list[Chunk]] = None, model_name: str = "all-MiniLM-L6-v2"):
        self.chunks: list[Chunk] = []
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None
        
        if chunks:
            self.add(chunks)
    
    def _load_model(self) -> None:
        """Lazy load the sentence transformer model."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
    
    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks and compute embeddings."""
        self._load_model()
        
        self.chunks.extend(chunks)
        texts = [c.text for c in self.chunks]
        
        self.embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.embeddings = self.embeddings / norms
    
    def query(self, query: str, k: int = 5) -> list[tuple[Chunk, float, int]]:
        """Query by cosine similarity, return list of (chunk, score, rank)."""
        if self.embeddings is None or self.model is None:
            return []
        
        # Embed query
        query_vec = self.model.encode([query], convert_to_numpy=True)[0]
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        
        # Compute similarities
        scores = np.dot(self.embeddings, query_vec)
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append((self.chunks[idx], float(scores[idx]), rank))
        
        return results


def load_dense(corpus_dir: Optional[Path] = None) -> DenseIndex:
    """Load or rebuild dense index from corpus."""
    if corpus_dir is None:
        corpus_dir = Path(__file__).parent.parent.parent / "corpus"
    
    from contextlab.chunking import chunk_corpus
    from contextlab.config import get_config
    
    config = get_config()
    chunks = chunk_corpus(
        corpus_dir,
        chunk_tokens=config["chunk_tokens"],
        chunk_overlap_tokens=config["chunk_overlap_tokens"],
        encoding=config["encoding"],
    )
    
    return DenseIndex(chunks, model_name=config["embedding_model"])

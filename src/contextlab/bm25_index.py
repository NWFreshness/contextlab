"""BM25 index for lexical retrieval — pure Python implementation."""
from pathlib import Path
from typing import Optional

import numpy as np
import tiktoken

from contextlab.types import Chunk


class BM25Index:
    """BM25 retriever with tiktoken tokenization."""
    
    def __init__(self, chunks: Optional[list[Chunk]] = None, k1: float = 1.5, b: float = 0.75):
        self.chunks: list[Chunk] = []
        self.k1 = k1
        self.b = b
        self.encoder = tiktoken.get_encoding("cl100k_base")
        
        # Index data
        self.tokenized_corpus: list[list[str]] = []
        self.doc_freqs: dict[str, int] = {}
        self.doc_lens: list[int] = []
        self.avgdl: float = 0.0
        self.N: int = 0
        self.idf: dict[str, float] = {}
        
        if chunks:
            self.add(chunks)
    
    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text using tiktoken."""
        tokens = self.encoder.encode(text)
        return [self.encoder.decode([t]) for t in tokens]
    
    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the index."""
        self.chunks.extend(chunks)
        texts = [c.text for c in self.chunks]
        
        # Tokenize corpus
        self.tokenized_corpus = [self._tokenize(t) for t in texts]
        self.doc_lens = [len(tokens) for tokens in self.tokenized_corpus]
        self.N = len(self.tokenized_corpus)
        self.avgdl = sum(self.doc_lens) / self.N if self.N > 0 else 0.0
        
        # Compute document frequencies
        self.doc_freqs = {}
        for doc_tokens in self.tokenized_corpus:
            for token in set(doc_tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
        
        # Compute IDF for all tokens
        self.idf = {}
        for token, df in self.doc_freqs.items():
            # Standard BM25 IDF formula
            self.idf[token] = np.log((self.N - df + 0.5) / (df + 0.5) + 1)
    
    def _score(self, query_tokens: list[str], doc_idx: int) -> float:
        """Compute BM25 score for a single document."""
        doc_tokens = self.tokenized_corpus[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        
        score = 0.0
        for token in query_tokens:
            if token not in self.idf:
                continue
            
            tf = doc_tokens.count(token)
            if tf == 0:
                continue
            
            idf = self.idf[token]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator
        
        return score
    
    def query(self, query: str, k: int = 5) -> list[tuple[Chunk, float, int]]:
        """Query BM25, return list of (chunk, score, rank)."""
        if not self.chunks or not self.tokenized_corpus:
            return []
        
        # Clamp k
        k = min(k, len(self.chunks))
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Score all documents
        scores = np.array([self._score(query_tokens, i) for i in range(self.N)])
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            if scores[idx] > 0:
                results.append((self.chunks[idx], float(scores[idx]), rank))
        
        return results
    
    @classmethod
    def load(cls, chunks_path: Path, index_dir: Path) -> "BM25Index":
        """Load index from disk."""
        chunks = []
        with open(chunks_path) as f:
            for line in f:
                chunks.append(Chunk.model_validate_json(line))
        
        return cls(chunks)


def load_bm25(corpus_dir: Optional[Path] = None) -> BM25Index:
    """Load or rebuild BM25 index from corpus."""
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
    
    return BM25Index(chunks)

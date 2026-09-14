"""Main retrieval module combining BM25, dense, and hybrid search."""
from pathlib import Path
from typing import Optional

from contextlab.bm25_index import BM25Index, load_bm25
from contextlab.dense_index import DenseIndex, load_dense
from contextlab.fuse import reciprocal_rank_fusion
from contextlab.config import get_config
from contextlab.trace import current_trace_id, start_span
from contextlab.types import Chunk, Hit, RetrieveRequest, RetrieveResponse


def ordered_unique(values) -> list:
    """De-duplicate while preserving first-seen order."""
    seen: dict = {}
    for value in values:
        seen.setdefault(value, None)
    return list(seen)


class Retriever:
    """Hybrid retrieval combining BM25 and dense search."""
    
    def __init__(
        self,
        corpus_dir: Optional[Path] = None,
        chunk_tokens: int = 256,
        chunk_overlap_tokens: int = 32,
        encoding: str = "cl100k_base",
        embedding_model: str = "all-MiniLM-L6-v2",
        bm25_top_n: int = 50,
        dense_top_n: int = 50,
        rrf_k: int = 60,
        hybrid_k: int = 5,
    ):
        self.corpus_dir = corpus_dir or Path(__file__).parent.parent.parent / "corpus"
        self.config = {
            "chunk_tokens": chunk_tokens,
            "chunk_overlap_tokens": chunk_overlap_tokens,
            "encoding": encoding,
            "embedding_model": embedding_model,
            "bm25_top_n": bm25_top_n,
            "dense_top_n": dense_top_n,
            "rrf_k": rrf_k,
            "hybrid_k": hybrid_k,
        }
        
        self.bm25 = load_bm25(self.corpus_dir)
        self.dense = load_dense(self.corpus_dir)
    
    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        """Run retrieval in specified mode.

        Each retriever hop gets a span (retrieve.bm25 / retrieve.dense /
        retrieve.fuse) parented to retrieve.<mode>; the span carries the chunk
        ids actually returned so a bad answer is reconstructable after the fact.
        """
        k = request.k
        mode = request.mode

        if mode == "bm25":
            with start_span("retrieve.bm25", query=request.query, mode="bm25", k=k) as span:
                bm25_results = self.bm25.query(request.query, k=k)
                hits = self._bm25_to_hits(bm25_results)
                span.set_attributes(
                    chunk_ids=[h.chunk_id for h in hits],
                    doc_ids=ordered_unique(h.doc_id for h in hits),
                    n_hits=len(hits),
                )
        elif mode == "dense":
            with start_span("retrieve.dense", query=request.query, mode="dense", k=k) as span:
                dense_results = self.dense.query(request.query, k=k)
                hits = self._dense_to_hits(dense_results)
                span.set_attributes(
                    chunk_ids=[h.chunk_id for h in hits],
                    doc_ids=ordered_unique(h.doc_id for h in hits),
                    n_hits=len(hits),
                )
        else:  # hybrid
            with start_span("retrieve.hybrid", query=request.query, mode="hybrid", k=request.k) as span:
                with start_span(
                    "retrieve.bm25", query=request.query, mode="bm25", k=self.config["bm25_top_n"]
                ) as bm25_span:
                    bm25_results = self.bm25.query(request.query, k=self.config["bm25_top_n"])
                    bm25_span.set_attributes(
                        chunk_ids=[chunk.chunk_id for chunk, _, _ in bm25_results],
                        doc_ids=ordered_unique(chunk.doc_id for chunk, _, _ in bm25_results),
                        n_hits=len(bm25_results),
                    )

                with start_span(
                    "retrieve.dense", query=request.query, mode="dense", k=self.config["dense_top_n"]
                ) as dense_span:
                    dense_results = self.dense.query(request.query, k=self.config["dense_top_n"])
                    dense_span.set_attributes(
                        chunk_ids=[chunk.chunk_id for chunk, _, _ in dense_results],
                        doc_ids=ordered_unique(chunk.doc_id for chunk, _, _ in dense_results),
                        n_hits=len(dense_results),
                    )

                with start_span(
                    "retrieve.fuse", query=request.query, mode="hybrid", k=self.config["hybrid_k"]
                ) as fuse_span:
                    hits = reciprocal_rank_fusion(
                        bm25_results,
                        dense_results,
                        rrf_k=self.config["rrf_k"],
                        k=self.config["hybrid_k"],
                    )
                    fuse_span.set_attributes(
                        chunk_ids=[h.chunk_id for h in hits],
                        doc_ids=ordered_unique(h.doc_id for h in hits),
                        n_hits=len(hits),
                        rrf_k=self.config["rrf_k"],
                    )

                span.set_attributes(
                    chunk_ids=[h.chunk_id for h in hits],
                    doc_ids=ordered_unique(h.doc_id for h in hits),
                    n_hits=len(hits),
                )

        settings = self.config.copy()
        trace_id = current_trace_id()
        if trace_id:
            settings["trace_id"] = trace_id

        return RetrieveResponse(
            query=request.query,
            mode=mode,
            hits=hits,
            settings=settings,
        )
    
    def _bm25_to_hits(self, results: list[tuple[Chunk, float, int]]) -> list[Hit]:
        """Convert BM25 results to Hit list."""
        hits = []
        for chunk, score, rank in results:
            citation = f"{chunk.doc_id} § {chunk.section}" if chunk.section else chunk.doc_id
            hits.append(Hit(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text[:200],
                score=score,
                scores={"bm25": score, "dense": 0.0, "rrf": 0.0},
                rank=rank,
                citation=citation,
            ))
        return hits
    
    def _dense_to_hits(self, results: list[tuple[Chunk, float, int]]) -> list[Hit]:
        """Convert dense results to Hit list."""
        hits = []
        for chunk, score, rank in results:
            citation = f"{chunk.doc_id} § {chunk.section}" if chunk.section else chunk.doc_id
            hits.append(Hit(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text[:200],
                score=score,
                scores={"bm25": 0.0, "dense": score, "rrf": 0.0},
                rank=rank,
                citation=citation,
            ))
        return hits


def retrieve(query: str, k: int = 5, mode: str = "hybrid") -> RetrieveResponse:
    """Convenience function for retrieval."""
    config = get_config()
    retriever = Retriever(
        chunk_tokens=config["chunk_tokens"],
        chunk_overlap_tokens=config["chunk_overlap_tokens"],
        encoding=config["encoding"],
        embedding_model=config["embedding_model"],
        bm25_top_n=config["bm25_top_n"],
        dense_top_n=config["dense_top_n"],
        rrf_k=config["rrf_k"],
        hybrid_k=config["hybrid_k"],
    )
    return retriever.retrieve(RetrieveRequest(query=query, k=k, mode=mode))


def _main() -> None:
    """`python -m contextlab.retrieve --query "E-4471" --k 5 --mode hybrid`.

    Delegates to the CLI entry point so the run is traced like every other hop.
    """
    import argparse

    from contextlab.cli import cmd_retrieve

    parser = argparse.ArgumentParser(description="ContextLab retrieval")
    parser.add_argument("--query", type=str, required=True, help="Query string")
    parser.add_argument("--k", type=int, default=5, help="Number of hits")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["bm25", "dense", "hybrid"])
    cmd_retrieve(parser.parse_args())


if __name__ == "__main__":
    _main()

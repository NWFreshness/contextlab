"""Ingestion: chunk corpus and persist to disk."""
import json
from pathlib import Path

import numpy as np

from contextlab.chunking import chunk_corpus
from contextlab.config import get_config
from contextlab.trace import start_trace


def ingest(corpus_dir: str | Path | None = None, data_dir: str | Path | None = None) -> list[dict]:
    """Ingest corpus, chunk files, persist chunks and embeddings.

    Traced as the "ingest" hop: the span records which documents produced which
    chunk ids, so a chunk id seen in a later retrieve span can be traced back to
    its source document.
    """
    with start_trace("ingest") as trace:
        config = get_config()

        if corpus_dir is None:
            corpus_dir = Path(__file__).parent.parent.parent / "corpus"
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data"

        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        # Chunk all documents
        chunks = chunk_corpus(
            corpus_dir,
            chunk_tokens=config["chunk_tokens"],
            chunk_overlap_tokens=config["chunk_overlap_tokens"],
            encoding=config["encoding"],
        )

        # Write chunks to JSONL
        chunks_path = data_dir / "chunks.jsonl"
        with open(chunks_path, "w") as f:
            for chunk in chunks:
                f.write(chunk.model_dump_json() + "\n")

        # Report chunk counts per doc
        from collections import Counter
        doc_counts = Counter(c.doc_id for c in chunks)
        print(f"Chunk counts per document:")
        for doc_id, count in sorted(doc_counts.items()):
            print(f"  {doc_id}: {count} chunks")
        print(f"Total: {len(chunks)} chunks")

        trace.set_attributes(
            n_docs=len(doc_counts),
            n_chunks=len(chunks),
            doc_ids=sorted(doc_counts),
            chunk_ids=[c.chunk_id for c in chunks],
            corpus_dir=str(corpus_dir),
            chunks_path=str(chunks_path),
        )

        return [c.model_dump() for c in chunks]


if __name__ == "__main__":
    ingest()

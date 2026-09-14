"""Evaluation script for retrieval on golden set."""
import json
from datetime import datetime, timezone
from pathlib import Path

from contextlab.retrieve import retrieve, Retriever
from contextlab.config import get_config


def evaluate(
    golden_path: str | Path = "evals/retrieval_golden.jsonl",
    output_path: str | Path = "artifacts/retrieval_eval.json",
) -> dict:
    """Run retrieval evaluation on golden set."""
    config = get_config()
    
    golden_path = Path(golden_path)
    output_path = Path(output_path)
    
    # Load golden cases
    cases = []
    with open(golden_path) as f:
        for line in f:
            cases.append(json.loads(line))
    
    # Initialize retriever
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
    
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "cases": [],
        "summary": {},
    }
    
    # Track per-mode recalls
    mode_recalls = {"bm25": 0, "dense": 0, "hybrid": 0}
    mode_counts = {"bm25": 0, "dense": 0, "hybrid": 0}
    
    for case in cases:
        case_result = {
            "id": case["id"],
            "query": case["query"],
            "intent": case["intent"],
            "relevant_chunk_ids": case["relevant_chunk_ids"],
            "relevant_doc_ids": case["relevant_doc_ids"],
            "hits": {},
        }
        
        for mode in ["bm25", "dense", "hybrid"]:
            response = retriever.retrieve(
                type("Request", (), {"query": case["query"], "k": 5, "mode": mode, "filters": None})()
            )
            
            hit_chunk_ids = [h.chunk_id for h in response.hits]
            
            # Compute recall@5
            relevant = set(case["relevant_chunk_ids"]) if case["relevant_chunk_ids"] else set(case["relevant_doc_ids"])
            hits_set = set(hit_chunk_ids)
            
            # Check if relevant chunk is in hits (doc-level fallback)
            if case["relevant_chunk_ids"]:
                recall = len(relevant & hits_set) / len(relevant) if relevant else 0.0
            else:
                # doc-level: check if any chunk from relevant doc is in hits
                hit_doc_ids = {h.split("::")[0] for h in hit_chunk_ids}
                relevant_docs = set(case["relevant_doc_ids"])
                recall = len(relevant_docs & hit_doc_ids) / len(relevant_docs) if relevant_docs else 0.0
            
            mode_recalls[mode] += recall
            mode_counts[mode] += 1
            
            case_result["hits"][mode] = {
                "hit_chunk_ids": hit_chunk_ids,
                "recall@5": recall,
            }
        
        results["cases"].append(case_result)
    
    # Compute aggregate recalls
    for mode in mode_recalls:
        results["summary"][f"recall@5_{mode}"] = mode_recalls[mode] / mode_counts[mode] if mode_counts[mode] else 0.0
    
    results["summary"]["num_cases"] = len(cases)
    
    # Print summary
    print(f"\n=== Retrieval Eval Results ===")
    print(f"Cases: {results['summary']['num_cases']}")
    print(f"BM25 recall@5:  {results['summary']['recall@5_bm25']:.3f}")
    print(f"Dense recall@5: {results['summary']['recall@5_dense']:.3f}")
    print(f"Hybrid recall@5: {results['summary']['recall@5_hybrid']:.3f}")
    print(f"\nHybrid must not be worse than max(BM25, Dense): ", end="")
    max_bm25_dense = max(results["summary"]["recall@5_bm25"], results["summary"]["recall@5_dense"])
    if results["summary"]["recall@5_hybrid"] >= max_bm25_dense - 0.001:
        print("PASS")
    else:
        print(f"FAIL (hybrid={results['summary']['recall@5_hybrid']:.3f} < max={max_bm25_dense:.3f})")
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {output_path}")
    
    return results


if __name__ == "__main__":
    evaluate()

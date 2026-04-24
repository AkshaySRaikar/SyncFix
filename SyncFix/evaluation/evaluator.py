# evaluation/evaluator.py
import time
from typing import List, Dict
from retrieval.pipeline import RetrievalPipeline

def evaluate_retrieval(
    pipeline: RetrievalPipeline,
    queries: List[Dict],   # [{"query": str, "relevant_ids": [str]}]
    k: int = 5,
) -> Dict:
    precisions, recalls, reciprocal_ranks, latencies = [], [], [], []

    for item in queries:
        query    = item["query"]
        relevant = set(item["relevant_ids"])

        t0 = time.perf_counter()
        results = pipeline.retrieve(query, top_k=k)
        latency = (time.perf_counter() - t0) * 1000

        retrieved_ids = [h["text"][:40] for h in results["hits"]]  # use chunk_id in practice
        retrieved_set = set(retrieved_ids)

        tp = len(relevant & retrieved_set)
        precisions.append(tp / k)
        recalls.append(tp / len(relevant) if relevant else 0)
        latencies.append(latency)

        # MRR
        rr = 0.0
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid in relevant:
                rr = 1 / rank
                break
        reciprocal_ranks.append(rr)

    return {
        f"Precision@{k}": round(sum(precisions) / len(precisions), 4),
        f"Recall@{k}":    round(sum(recalls) / len(recalls), 4),
        "MRR":            round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4),
        "Avg latency ms": round(sum(latencies) / len(latencies), 1),
    }
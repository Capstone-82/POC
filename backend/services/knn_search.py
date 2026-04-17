"""
KNN similarity search and per-model signal aggregation for semantic routing.
"""
from __future__ import annotations

from statistics import median, stdev
from typing import Any


MIN_NEIGHBOR_SIMILARITY = 0.60   # lowered from 0.72 until 1536-dim backfill completes
DEFAULT_K = 20
FALLBACK_K = 40
AGGREGATION_FALLBACK_K = 240
FALLBACK_SIMILARITY = 0.50
MIN_MODEL_NEIGHBORS = 1          # lowered from 3 until backfill gives more neighbors per model


def search_neighbors(
    supabase_client: Any,
    embedding: list[float],
    use_case: str,
    k: int = DEFAULT_K,
    min_similarity: float = MIN_NEIGHBOR_SIMILARITY,
) -> list[dict]:
    """
    Call the Supabase knn_search RPC.
    """
    result = supabase_client.rpc(
        "knn_search",
        {
            "query_embedding": embedding,
            "target_use_case": use_case,
            "result_limit": k,
            "min_similarity": min_similarity,
        },
    ).execute()
    return result.data or []


def aggregate_knn_signals(neighbors: list[dict]) -> dict[str, dict]:
    """
    Aggregate neighboring benchmark rows into model-level routing signals.
    """
    grouped: dict[str, list[dict]] = {}
    for row in neighbors:
        model_id = row["model_id"]
        grouped.setdefault(model_id, []).append(row)

    aggregated: dict[str, dict] = {}
    for model_id, rows in grouped.items():
        if len(rows) < MIN_MODEL_NEIGHBORS:
            continue

        sims = [float(row["similarity"]) for row in rows]
        accs = [float(row["avg_accuracy_score"]) for row in rows]
        costs = [float(row["cost"]) for row in rows]
        latencies = [float(row["latency_ms"]) for row in rows]

        total_sim = sum(sims)
        if total_sim <= 0:
            continue

        sim_weighted_accuracy = sum(sim * acc for sim, acc in zip(sims, accs)) / total_sim

        aggregated[model_id] = {
            "model_id": model_id,
            "provider": rows[0].get("provider", ""),
            "sim_weighted_accuracy": round(sim_weighted_accuracy, 2),
            "p50_cost": round(median(costs), 6),
            "p50_latency": round(median(latencies), 1),
            "score_variance": round(stdev(accs) if len(accs) > 1 else 0, 2),
            "sample_n": len(rows),
        }

    return aggregated

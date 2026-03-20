from services.supabase_client import get_benchmark_data

WEIGHTS = {"accuracy": 0.6, "cost": 0.3, "latency": 0.1}


async def get_recommendation(
    use_case: str,
    complexity: str,
    quality_score: int,
    current_model: str,
) -> dict:
    """
    Recommend the best model based on benchmark data.

    Pulls rows filtered by use_case + complexity, aggregates per-model averages,
    applies weighted composite scoring, and returns deltas vs. user's current model.
    """

    # Pull benchmark rows for this use_case + complexity
    rows = await get_benchmark_data(use_case=use_case, complexity=complexity)

    if not rows:
        raise ValueError("Not enough benchmark data for this use case and complexity.")

    # Aggregate per model
    model_stats = {}
    for row in rows:
        mid = row["model_id"]
        if mid not in model_stats:
            model_stats[mid] = {
                "accuracy_scores": [],
                "costs": [],
                "latencies": [],
                "provider": row["provider"],
            }
        model_stats[mid]["accuracy_scores"].append(row["accuracy_score"])
        model_stats[mid]["costs"].append(row["cost"])
        model_stats[mid]["latencies"].append(row["latency_ms"])

    # Compute averages + composite score
    scored = []
    for model_id, stats in model_stats.items():
        avg_acc = sum(stats["accuracy_scores"]) / len(stats["accuracy_scores"])
        avg_cost = sum(stats["costs"]) / len(stats["costs"])
        avg_lat = sum(stats["latencies"]) / len(stats["latencies"])

        scored.append({
            "model_id": model_id,
            "provider": stats["provider"],
            "avg_accuracy": avg_acc,
            "avg_cost": avg_cost,
            "avg_latency": avg_lat,
        })

    # Normalize and score
    max_acc = max(s["avg_accuracy"] for s in scored)
    min_acc = min(s["avg_accuracy"] for s in scored)
    max_cost = max(s["avg_cost"] for s in scored)
    min_cost = min(s["avg_cost"] for s in scored)
    max_lat = max(s["avg_latency"] for s in scored)
    min_lat = min(s["avg_latency"] for s in scored)

    def norm(val, lo, hi, invert=False):
        if hi == lo:
            return 1.0
        n = (val - lo) / (hi - lo)
        return 1 - n if invert else n

    for s in scored:
        s["composite"] = (
            WEIGHTS["accuracy"] * norm(s["avg_accuracy"], min_acc, max_acc)
            + WEIGHTS["cost"] * norm(s["avg_cost"], min_cost, max_cost, invert=True)
            + WEIGHTS["latency"] * norm(s["avg_latency"], min_lat, max_lat, invert=True)
        )

    scored.sort(key=lambda x: x["composite"], reverse=True)
    best = scored[0]

    # Find current model stats
    current = next((s for s in scored if s["model_id"] == current_model), None)
    if not current:
        current = {"avg_accuracy": 0, "avg_cost": 0, "avg_latency": 0}

    acc_delta = round(best["avg_accuracy"] - current["avg_accuracy"], 1)
    cost_delta = round(
        ((best["avg_cost"] - current["avg_cost"]) / max(current["avg_cost"], 0.0001)) * 100,
        1,
    )
    latency_delta = int(best["avg_latency"] - current["avg_latency"])

    reason = (
        f"{complexity.capitalize()}-complexity {use_case} prompt. "
        f"{best['model_id']} outperforms {current_model} on this task type "
        f"in our benchmark data."
    )

    return {
        "recommended_model": best["model_id"],
        "accuracy_delta": acc_delta,
        "cost_delta": cost_delta,
        "latency_delta": latency_delta,
        "reason": reason,
    }

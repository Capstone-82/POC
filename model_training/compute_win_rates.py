from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from generate_avg_accuracy_scores import get_supabase_client


MIN_DECISIVE_MATCHES = 5
FULL_CONFIDENCE_MATCHES = 10


def _add_judges(bucket: dict, judge_model: str) -> None:
    for judge in judge_model.split(","):
        cleaned = judge.strip()
        if cleaned:
            bucket["judges"].add(cleaned)


def _update_bucket(bucket: dict, model_id: str, winner: str, winner_model: str, judge_model: str) -> None:
    bucket["total_matches"] += 1
    bucket["total_participations"] += 1

    if judge_model:
        _add_judges(bucket, judge_model)

    if winner == "TIE":
        bucket["ties"] += 1
    elif winner_model == model_id:
        bucket["wins"] += 1
    else:
        bucket["losses"] += 1


def refresh_model_win_rates(supabase) -> int:
    page_size = 1000
    start = 0
    rows: list[dict] = []

    while True:
        batch = (
            supabase.table("pairwise_results")
            .select("use_case,complexity,model_a,model_b,winner,winner_model,judge_model")
            .range(start, start + page_size - 1)
            .execute()
            .data
            or []
        )
        if not batch:
            break
        rows.extend(batch)
        start += page_size

    aggregate: dict[tuple[str, str, str], dict] = defaultdict(
        lambda: {
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "total_matches": 0,
            "total_participations": 0,
            "judges": set(),
        }
    )

    for row in rows:
        use_case = str(row.get("use_case", "")).strip()
        complexity = str(row.get("complexity", "")).strip().lower() or "all"
        winner = str(row.get("winner", "")).strip().upper()
        winner_model = str(row.get("winner_model", "")).strip()
        model_a = str(row.get("model_a", "")).strip()
        model_b = str(row.get("model_b", "")).strip()
        judge_model = str(row.get("judge_model", "")).strip()

        for model_id in (model_a, model_b):
            if not model_id or not use_case:
                continue

            bucket = aggregate[(model_id, use_case, complexity)]
            _update_bucket(bucket, model_id, winner, winner_model, judge_model)

            all_bucket = aggregate[(model_id, use_case, "all")]
            _update_bucket(all_bucket, model_id, winner, winner_model, judge_model)

    upserts = []
    for (model_id, use_case, complexity), bucket in aggregate.items():
        decisive_matches = bucket["wins"] + bucket["losses"]
        confidence = min(1.0, decisive_matches / FULL_CONFIDENCE_MATCHES) if decisive_matches else 0.0
        tie_rate = (bucket["ties"] / bucket["total_matches"]) if bucket["total_matches"] else 0.0

        # Guardrail: do not emit a usable win_rate for very sparse slices.
        win_rate = None
        if decisive_matches >= MIN_DECISIVE_MATCHES:
            win_rate = bucket["wins"] / decisive_matches

        upserts.append(
            {
                "model_id": model_id,
                "use_case": use_case,
                "complexity": complexity,
                "win_rate": None if win_rate is None else round(win_rate, 4),
                "total_matches": bucket["total_matches"],
                "total_participations": bucket["total_participations"],
                "decisive_matches": decisive_matches,
                "wins": bucket["wins"],
                "losses": bucket["losses"],
                "ties": bucket["ties"],
                "tie_rate": round(tie_rate, 4),
                "judge_count": len(bucket["judges"]),
                "confidence": round(confidence, 4),
            }
        )

    if upserts:
        supabase.table("model_win_rates").upsert(
            upserts,
            on_conflict="model_id,use_case,complexity",
        ).execute()
    return len(upserts)


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / "backend" / ".env")
    supabase = get_supabase_client()
    rows_written = refresh_model_win_rates(supabase)
    print(f"Refreshed {rows_written} model_win_rates rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

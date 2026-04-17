import os
from typing import Any, Dict, List, Optional
from supabase import create_client

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY"),
)


async def save_row(row: dict):
    """Insert a single benchmark result row into Supabase and warm prompt metadata when possible."""
    payload = dict(row)
    prompt = str(payload.get("prompt", "") or "").strip()

    if prompt and not payload.get("prompt_hash"):
        try:
            from services.embedding_service import compute_prompt_hash

            payload["prompt_hash"] = compute_prompt_hash(prompt)
        except Exception:
            pass

    supabase.table("benchmark_results").insert(payload).execute()

    if prompt:
        try:
            from services.embedding_service import get_or_compute_embedding

            await get_or_compute_embedding(prompt, supabase)
        except Exception:
            pass


async def save_prompt_log(log: dict):
    """
    Insert a prompt log into the prompt_logs table.
    Preferred columns: prompt_hash, prompt, use_case, clarity.
    Falls back to the legacy schema if prompt_hash is not available yet.
    """
    try:
        supabase.table("prompt_logs").insert(log).execute()
    except Exception:
        legacy_payload = {
            key: value
            for key, value in log.items()
            if key in {"prompt", "use_case", "clarity"}
        }
        supabase.table("prompt_logs").insert(legacy_payload).execute()


def _fetch_all(query, page_size: int = 1000) -> List[dict]:
    rows: List[dict] = []
    start = 0

    while True:
        response = query.range(start, start + page_size - 1).execute()
        batch = response.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size

    return rows


async def get_benchmark_data(
    use_case: Optional[str] = None,
    complexity: Optional[str] = None,
    clarity: Optional[str] = None,
) -> List[dict]:
    """Query benchmark results with optional filters."""
    columns = (
        "id,model_id,provider,use_case,prompt_complexity,clarity,"
        "accuracy_score,avg_accuracy_score,cost,latency_ms,prompt,prompt_hash,"
        "response,syntax_pass,syntax_checked,consistency_score,win_rate,is_correct"
    )
    query = supabase.table("benchmark_results").select(columns)
    if use_case:
        query = query.eq("use_case", use_case)
    if complexity:
        query = query.eq("prompt_complexity", complexity)
    if clarity:
        query = query.eq("clarity", clarity)
    return _fetch_all(query)


async def get_prompt_logs(use_case: Optional[str] = None, prompt: Optional[str] = None) -> List[dict]:
    """Query prompt logs with optional exact-match filters."""
    query = supabase.table("prompt_logs").select("*")
    if use_case:
        query = query.eq("use_case", use_case)
    if prompt:
        query = query.eq("prompt", prompt)
    return _fetch_all(query)


async def get_model_win_rates(
    use_case: str,
    complexity: Optional[str] = None,
    min_matches: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    Return model-level pairwise routing stats for the requested use-case slice.
    Falls back from a specific complexity bucket to 'all' if needed.
    """
    # Defensively cast use_case to string to prevent Enum serialization bugs in Supabase
    use_case = use_case.value if hasattr(use_case, "value") else str(use_case)
    target_complexity = (complexity or "all").strip().lower()

    def _query(slice_complexity: str) -> List[dict]:
        return (
            supabase.table("model_win_rates")
            .select("model_id,win_rate,total_matches,decisive_matches,confidence,tie_rate")
            .eq("use_case", use_case)
            .eq("complexity", slice_complexity)
            .not_.is_("win_rate", "null")
            .gte("decisive_matches", min_matches)
            .execute()
            .data
            or []
        )

    rows = _query(target_complexity)
    if not rows and target_complexity != "all":
        rows = _query("all")

    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        model_id = str(row.get("model_id", "")).strip()
        try:
            win_rate = float(row.get("win_rate"))
        except (TypeError, ValueError):
            continue
        if model_id:
            result[model_id] = {
                "win_rate": win_rate,
                "confidence": float(row.get("confidence") or 0.0),
                "decisive_matches": int(row.get("decisive_matches") or 0),
                "total_matches": int(row.get("total_matches") or 0),
                "tie_rate": float(row.get("tie_rate") or 0.0),
            }
    return result

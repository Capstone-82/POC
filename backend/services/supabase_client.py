import os
from typing import List, Optional
from supabase import create_client

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY"),
)


async def save_row(row: dict):
    """Insert a single benchmark result row into Supabase."""
    supabase.table("benchmark_results").insert(row).execute()


async def save_prompt_log(log: dict):
    """
    Insert a prompt log into the prompt_logs table.
    Columns: prompt, use_case, clarity
    """
    supabase.table("prompt_logs").insert(log).execute()


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
    query = supabase.table("benchmark_results").select("*")
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

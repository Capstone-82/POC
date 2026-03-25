import os
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


async def get_benchmark_data(use_case: str = None, complexity: str = None) -> list[dict]:
    """Query benchmark results with optional filters."""
    query = supabase.table("benchmark_results").select("*")
    if use_case:
        query = query.eq("use_case", use_case)
    if complexity:
        query = query.eq("prompt_complexity", complexity)
    result = query.execute()
    return result.data

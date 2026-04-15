"""
Backfill prompt_hash on benchmark_results rows.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client


PAGE_SIZE = 500


def compute_prompt_hash(prompt: str) -> str:
    normalized = re.sub(r"\s+", " ", (prompt or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / "backend" / ".env")
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

    start = 0
    updated = 0
    while True:
        rows = (
            supabase.table("benchmark_results")
            .select("id,prompt")
            .is_("prompt_hash", "null")
            .range(start, start + PAGE_SIZE - 1)
            .execute()
            .data
            or []
        )
        if not rows:
            break

        for row in rows:
            prompt_hash = compute_prompt_hash(str(row.get("prompt", "")))
            supabase.table("benchmark_results").update({"prompt_hash": prompt_hash}).eq("id", row["id"]).execute()
            updated += 1

        print(f"Updated: {updated}")

    print("Prompt hash backfill complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

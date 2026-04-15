"""
One-time script to compute and store embeddings for benchmark prompt hashes.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client


PAGE_SIZE = 500
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / "backend" / ".env")
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    model = SentenceTransformer(MODEL_NAME)

    existing = {
        row["prompt_hash"]
        for row in (supabase.table("prompt_embeddings").select("prompt_hash").execute().data or [])
        if row.get("prompt_hash")
    }

    start = 0
    processed = 0
    while True:
        rows = (
            supabase.table("benchmark_results")
            .select("prompt_hash,prompt")
            .not_.is_("prompt_hash", "null")
            .range(start, start + PAGE_SIZE - 1)
            .execute()
            .data
            or []
        )
        if not rows:
            break

        unique_to_embed = []
        seen_in_page = set()
        for row in rows:
            prompt_hash = row.get("prompt_hash")
            if not prompt_hash or prompt_hash in existing or prompt_hash in seen_in_page:
                continue
            unique_to_embed.append(row)
            seen_in_page.add(prompt_hash)

        if unique_to_embed:
            texts = [str(row.get("prompt", "")) for row in unique_to_embed]
            vectors = model.encode(texts, batch_size=64, show_progress_bar=True).tolist()
            for row, vector in zip(unique_to_embed, vectors):
                supabase.table("prompt_embeddings").upsert(
                    {
                        "prompt_hash": row["prompt_hash"],
                        "embedding": vector,
                        "model_name": "all-MiniLM-L6-v2",
                    },
                    on_conflict="prompt_hash",
                ).execute()
                existing.add(row["prompt_hash"])

        start += PAGE_SIZE
        processed += len(rows)
        print(f"Processed: {processed}")

    print("Embedding backfill complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Batch re-embed all benchmark_results rows using OpenAI text-embedding-3-small.

Run this once after switching from sentence-transformers (MiniLM-L6-v2) to the
OpenAI embedding model.  Old MiniLM vectors are in a completely different vector
space — cosine similarity between them and OpenAI vectors is near 0, which is
why KNN returns 0 usable neighbors.

Usage (from backend/ directory with .env present):
    python ../model_training/backfill_openai_embeddings.py

What it does:
    1. Truncates the prompt_embeddings table (removes all stale MiniLM vectors).
    2. Fetches every distinct (prompt_hash, prompt) pair from benchmark_results.
    3. Calls OpenAI text-embedding-3-small (dimensions=384) in batches of 100.
    4. Upserts results into prompt_embeddings.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from supabase import create_client

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).resolve().parents[1] / "backend" / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
OPENAI_KEY   = os.getenv("OPENAI_API_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    sys.exit("ERROR: SUPABASE_URL / SUPABASE_KEY not set in .env")
if not OPENAI_KEY:
    sys.exit("ERROR: OPENAI_API_KEY not set in .env")

OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"
EMBED_MODEL      = "text-embedding-3-small"
EMBED_DIM        = 384
BATCH_SIZE       = 50      # OpenAI allows up to 2048 inputs per request
PAGE_SIZE        = 500     # rows per Supabase page
RATE_LIMIT_SLEEP = 0.3     # seconds between OpenAI batches


# ── OpenAI embedding ──────────────────────────────────────────────────────────

async def embed_batch(texts: list[str], client: httpx.AsyncClient) -> list[list[float]]:
    """Call OpenAI embeddings API for a batch of texts. Returns list of vectors."""
    response = await client.post(
        OPENAI_EMBED_URL,
        headers={
            "Authorization": f"Bearer {OPENAI_KEY}",
            "Content-Type":  "application/json",
        },
        json={
            "input":      texts,
            "model":      EMBED_MODEL,
            "dimensions": EMBED_DIM,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()["data"]
    # OpenAI returns items sorted by index
    data.sort(key=lambda x: x["index"])
    return [[float(v) for v in item["embedding"]] for item in data]


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Step 1: Truncate stale MiniLM embeddings
    print("Step 1: Clearing old embeddings from prompt_embeddings table...")
    try:
        # Delete all rows (Supabase REST doesn't support TRUNCATE, use delete with filter that always matches)
        supabase.table("prompt_embeddings").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        print("  ✓ Cleared old embeddings")
    except Exception as exc:
        print(f"  ✗ Clear failed: {exc}")
        print("  → Run manually in Supabase SQL editor: TRUNCATE prompt_embeddings;")

    # Step 2: Collect all unique (prompt_hash, prompt) pairs
    print("\nStep 2: Fetching unique prompt hashes from benchmark_results...")
    seen_hashes: set[str] = set()
    pairs: list[tuple[str, str]] = []  # (hash, prompt)
    offset = 0

    while True:
        rows = (
            supabase.table("benchmark_results")
            .select("prompt_hash,prompt")
            .not_.is_("prompt_hash", "null")
            .range(offset, offset + PAGE_SIZE - 1)
            .execute()
            .data or []
        )
        if not rows:
            break
        for row in rows:
            ph = row.get("prompt_hash", "").strip()
            pm = (row.get("prompt") or "").strip()
            if ph and pm and ph not in seen_hashes:
                seen_hashes.add(ph)
                pairs.append((ph, pm))
        offset += PAGE_SIZE
        print(f"  Fetched {offset} rows, unique hashes so far: {len(pairs)}")

    if not pairs:
        print("  No rows with prompt_hash found in benchmark_results.")
        print("  → Run backfill_prompt_hash.py first to compute prompt hashes.")
        return

    print(f"  ✓ Found {len(pairs)} unique prompts to embed")

    # Step 3: Embed in batches and upsert
    print(f"\nStep 3: Embedding {len(pairs)} prompts with {EMBED_MODEL} (dim={EMBED_DIM})...")
    total_batches = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE
    embedded = 0
    failed   = 0

    async with httpx.AsyncClient() as http_client:
        for batch_idx in range(0, len(pairs), BATCH_SIZE):
            batch_pairs  = pairs[batch_idx : batch_idx + BATCH_SIZE]
            batch_hashes = [p[0] for p in batch_pairs]
            batch_texts  = [p[1] for p in batch_pairs]

            try:
                vectors = await embed_batch(batch_texts, http_client)
                records = [
                    {"prompt_hash": ph, "embedding": vec}
                    for ph, vec in zip(batch_hashes, vectors)
                ]
                supabase.table("prompt_embeddings").upsert(
                    records, on_conflict="prompt_hash"
                ).execute()
                embedded += len(records)
            except Exception as exc:
                failed += len(batch_pairs)
                print(f"  ✗ Batch {batch_idx // BATCH_SIZE + 1}/{total_batches} failed: {exc}")
                continue

            done_batches = batch_idx // BATCH_SIZE + 1
            print(f"  Batch {done_batches}/{total_batches} — embedded {embedded}, failed {failed}")
            await asyncio.sleep(RATE_LIMIT_SLEEP)

    print(f"\n✓ Done. Embedded: {embedded} | Failed: {failed}")
    if failed:
        print("  Re-run the script to retry failed batches (will skip already-embedded hashes).")


if __name__ == "__main__":
    asyncio.run(main())

"""
backfill_1536_embeddings.py
============================
Fills prompt_embeddings.embedding (vector(1536)) for rows still NULL.
Safe to re-run — only processes rows where embedding IS NULL.

Run from: C:\\Users\\Musharraf\\Documents\\POC\\model_training\\
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / "backend" / ".env")

import httpx
from supabase import create_client

SUPABASE_URL   = os.getenv("SUPABASE_URL")
SUPABASE_KEY   = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL    = "text-embedding-3-small"
EMBED_DIM      = 1536
BATCH_SIZE     = 25          # smaller than before to reduce failure surface
CONCURRENCY    = 3           # concurrent batches

assert SUPABASE_URL and SUPABASE_KEY, "SUPABASE_URL / SUPABASE_KEY not set"
assert OPENAI_API_KEY, "OPENAI_API_KEY not set"

sb = create_client(SUPABASE_URL, SUPABASE_KEY)


# ── Supabase helpers ──────────────────────────────────────────────────────────

def _fetch_all(query, page_size: int = 1000) -> list[dict]:
    rows, start = [], 0
    while True:
        batch = query.range(start, start + page_size - 1).execute().data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
    return rows


# ── OpenAI embed with retry ───────────────────────────────────────────────────

async def _call_openai(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={"input": texts, "model": EMBED_MODEL, "dimensions": EMBED_DIM},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]


async def embed_with_retry(
    texts: list[str],
    max_retries: int = 4,
    base_delay: float = 3.0,
) -> list[list[float]] | None:
    """Exponential back-off retry around the OpenAI call. Returns None on total failure."""
    for attempt in range(max_retries):
        try:
            return await _call_openai(texts)
        except Exception as exc:
            wait = base_delay * (2 ** attempt)
            print(f"    [retry {attempt+1}/{max_retries}] {type(exc).__name__}: {exc} — wait {wait:.0f}s")
            await asyncio.sleep(wait)
    return None


# ── Per-batch processor with split-on-failure ─────────────────────────────────

async def process_batch(
    sem: asyncio.Semaphore,
    rows: list[dict],
    done: list[int],
    failed: list[str],
    total: int,
) -> None:
    """
    Embed + upsert. On embed failure, splits the batch in half and
    retries recursively down to 1-row batches before recording failure.
    """
    if not rows:
        return

    async with sem:
        vectors = await embed_with_retry([r["prompt"] for r in rows])

    if vectors is None:
        if len(rows) == 1:
            ph = rows[0]["prompt_hash"]
            print(f"  [FAIL] single row abandoned: {ph[:20]}")
            failed.append(ph)
            return
        # Split in half and retry each half
        mid = len(rows) // 2
        print(f"  [SPLIT] {len(rows)} rows → {mid} + {len(rows)-mid}")
        await process_batch(sem, rows[:mid],  done, failed, total)
        await process_batch(sem, rows[mid:],  done, failed, total)
        return

    # Upsert with per-row retry
    for row, vec in zip(rows, vectors):
        ph = row["prompt_hash"]
        for attempt in range(3):
            try:
                sb.table("prompt_embeddings") \
                  .update({"embedding": vec}) \
                  .eq("prompt_hash", ph) \
                  .execute()
                break
            except Exception as exc:
                if attempt == 2:
                    print(f"  [FAIL] upsert {ph[:20]}: {exc}")
                    failed.append(ph)
                else:
                    await asyncio.sleep(1.5 * (attempt + 1))

    done[0] += len(rows)
    pct = done[0] / total * 100
    print(f"  [{done[0]}/{total} — {pct:.0f}%]  batch of {len(rows)} done")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("\n" + "="*60)
    print("  Backfill prompt_embeddings.embedding (1536-dim)")
    print("="*60 + "\n")

    # 1. Find prompt_hashes that still need embedding
    print("Fetching rows with NULL embedding ...")
    pe_rows = _fetch_all(
        sb.table("prompt_embeddings")
          .select("prompt_hash")
          .is_("embedding", "null")
    )
    if not pe_rows:
        print("✓ All rows already have embedding. Nothing to do.")
        return

    missing_hashes = {r["prompt_hash"] for r in pe_rows if r.get("prompt_hash")}
    print(f"  {len(missing_hashes)} hashes still missing embedding")

    # 2. Fetch the prompt text for those hashes from benchmark_results
    print("Fetching prompt text from benchmark_results ...")
    br_rows = _fetch_all(
        sb.table("benchmark_results").select("prompt_hash, prompt")
    )
    # Deduplicate: one entry per prompt_hash
    prompt_map: dict[str, str] = {}
    for r in br_rows:
        ph = r.get("prompt_hash")
        pt = (r.get("prompt") or "").strip()
        if ph and pt and ph not in prompt_map:
            prompt_map[ph] = pt

    all_rows = [
        {"prompt_hash": ph, "prompt": prompt_map[ph]}
        for ph in missing_hashes
        if ph in prompt_map and prompt_map[ph]
    ]

    no_prompt = missing_hashes - {r["prompt_hash"] for r in all_rows}
    if no_prompt:
        print(f"  [WARN] {len(no_prompt)} hashes have no prompt text in benchmark_results — skipping")

    total = len(all_rows)
    if total == 0:
        print("No actionable rows found. Exiting.")
        return

    print(f"  {total} rows to embed and upsert\n")

    # 3. Process in batches
    batches = [all_rows[i : i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    sem     = asyncio.Semaphore(CONCURRENCY)
    done    = [0]
    failed: list[str] = []
    t0      = time.time()

    # Run sequentially within each semaphore slot (gather manages concurrency)
    await asyncio.gather(*[
        process_batch(sem, batch, done, failed, total)
        for batch in batches
    ])

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Done: {done[0]}/{total} embedded in {elapsed:.1f}s")
    if failed:
        print(f"  FAILED ({len(failed)} rows): {failed}")
    else:
        print("  All rows succeeded ✓")
    print("="*60)

    if failed:
        # Write failed hashes to file for manual inspection
        fail_path = Path(__file__).parent / "backfill_failed.txt"
        fail_path.write_text("\n".join(failed))
        print(f"\n  Failed hashes written to: {fail_path}")

    remaining = _fetch_all(
        sb.table("prompt_embeddings")
          .select("prompt_hash", count="exact")
          .is_("embedding", "null")
    )
    print(f"\n  Remaining NULL embedding rows: {len(remaining)}")

    if len(remaining) == 0:
        print("\nAll rows have embeddings. KNN search is fully operational.")


if __name__ == "__main__":
    asyncio.run(main())

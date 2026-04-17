"""
Prompt embedding service — OpenAI text-embedding-3-small (async HTTP via httpx).

Why OpenAI instead of sentence-transformers / fastembed / ONNX:
  All local embedding libraries (PyTorch, ONNX Runtime) fail on this Windows
  machine with [WinError 1114] DLL initialisation failure when imported inside
  uvicorn's async worker process.  The error is a Windows-specific interaction
  between asyncio thread pool workers and native DLL loading (CUDA / MKL state).
  It does NOT reproduce in a plain python -c "..." call but DOES reproduce inside
  the server worker — making it very hard to fix at the OS level.

  OpenAI text-embedding-3-small:
    - Pure HTTP request via httpx (already installed, zero native DLLs).
    - dimensions=384 → same vector size as all-MiniLM-L6-v2.
    - No schema change needed for the prompt_embeddings table (vector(384)).
    - Works from any async context without DLL issues.
    - ~0.02 USD / 1M tokens (cheap; costs only on cache misses).
    - OPENAI_API_KEY is already in .env.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

import httpx


OPENAI_EMBED_URL   = "https://api.openai.com/v1/embeddings"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMENSIONS   = 1536  # matches vector(1536) in prompt_embeddings table


def _get_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY env var is not set")
    return key


# ─── Core helpers ─────────────────────────────────────────────────────────────

def compute_prompt_hash(prompt: str) -> str:
    """SHA-256 of lowercased, whitespace-normalised prompt (first 32 hex chars)."""
    normalized = re.sub(r"\s+", " ", (prompt or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


async def embed_text_async(text: str) -> list[float]:
    """
    Call OpenAI text-embedding-3-small and return a 1536-dim vector.
    Uses httpx.AsyncClient for non-blocking HTTP inside the async event loop.
    """
    api_key = _get_api_key()
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            OPENAI_EMBED_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "input":      text,
                "model":      OPENAI_EMBED_MODEL,
                "dimensions": EMBED_DIMENSIONS,
            },
        )
        response.raise_for_status()
        data = response.json()
        return [float(v) for v in data["data"][0]["embedding"]]


# ─── Supabase vector coercion ─────────────────────────────────────────────────

def _coerce_embedding(value: Any) -> list[float] | None:
    """
    Supabase may return the pgvector column as a Python list, a JSON string,
    or a Postgres-formatted string like "[0.1,0.2,...]".  Handle all three.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
        except json.JSONDecodeError:
            pass
        if text.startswith("[") and text.endswith("]"):
            return [float(x.strip()) for x in text[1:-1].split(",") if x.strip()]
    return None


# ─── Public API ───────────────────────────────────────────────────────────────

async def get_or_compute_embedding(
    prompt: str,
    supabase_client: Any,
) -> tuple[list[float], str, bool]:
    """
    Return (embedding_vector, prompt_hash, was_cached).

    1. Hash the prompt (cheap, pure Python).
    2. Check Supabase prompt_embeddings cache.
    3. Cache hit  → return vector immediately (no OpenAI call).
    4. Cache miss → call OpenAI, store result, return vector.
    """
    prompt_hash = compute_prompt_hash(prompt)

    # ── Cache lookup ──────────────────────────────────────────────────────────
    try:
        result = (
            supabase_client.table("prompt_embeddings")
            .select("embedding")
            .eq("prompt_hash", prompt_hash)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        if rows and rows[0].get("embedding"):
            cached = _coerce_embedding(rows[0]["embedding"])
            if cached:
                return cached, prompt_hash, True
    except Exception as cache_exc:
        print(f"[EMBEDDING CACHE READ ERROR] {cache_exc}")

    # ── Compute via OpenAI + store ────────────────────────────────────────────
    vector = await embed_text_async(prompt)

    try:
        supabase_client.table("prompt_embeddings").upsert(
            {"prompt_hash": prompt_hash, "embedding": vector},
            on_conflict="prompt_hash",
        ).execute()
    except Exception as write_exc:
        print(f"[EMBEDDING CACHE WRITE ERROR] {write_exc}")

    return vector, prompt_hash, False

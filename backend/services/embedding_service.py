"""
Prompt embedding service backed by sentence-transformers and Supabase caching.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def compute_prompt_hash(prompt: str) -> str:
    normalized = re.sub(r"\s+", " ", (prompt or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def embed_text(text: str) -> list[float]:
    """Compute a 384-dimensional embedding for a string."""
    return _get_model().encode(text, convert_to_numpy=True).tolist()


def _coerce_embedding(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [float(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [float(item) for item in parsed]
        except json.JSONDecodeError:
            pass
        if text.startswith("[") and text.endswith("]"):
            return [float(item.strip()) for item in text[1:-1].split(",") if item.strip()]
    return None


async def get_or_compute_embedding(prompt: str, supabase_client: Any) -> tuple[list[float], str, bool]:
    """
    Return (embedding_vector, prompt_hash, was_cached), writing cache misses to
    the prompt_embeddings table.
    """
    prompt_hash = compute_prompt_hash(prompt)

    result = (
        supabase_client.table("prompt_embeddings")
        .select("embedding")
        .eq("prompt_hash", prompt_hash)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if rows and rows[0].get("embedding"):
        cached_vector = _coerce_embedding(rows[0]["embedding"])
        if cached_vector:
            return cached_vector, prompt_hash, True

    vector = embed_text(prompt)
    supabase_client.table("prompt_embeddings").upsert(
        {
            "prompt_hash": prompt_hash,
            "embedding": vector,
        },
        on_conflict="prompt_hash",
    ).execute()

    return vector, prompt_hash, False

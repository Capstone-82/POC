"""
GCP Vertex AI model service.
Accesses Gemini and other Vertex AI text models via the google-genai SDK.
"""

import time
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from google import genai

client = genai.Client(
    vertexai=True,
    api_key=os.getenv("GOOGLE_API_KEY"),
)

# ─── Vertex AI Model Registry ─────────────────────────────────
VERTEX_MODELS = [
    {
        "model_id": "gemini-2.5-pro",
        "provider": "Google",
        "short_id": "gemini-2-5-pro",
    },
    {
        "model_id": "gemini-2.5-flash",
        "provider": "Google",
        "short_id": "gemini-2-5-flash",
    },
    {
        "model_id": "gemini-2.0-flash",
        "provider": "Google",
        "short_id": "gemini-2-0-flash",
    },
    {
        "model_id": "gemini-2.0-flash-lite",
        "provider": "Google",
        "short_id": "gemini-2-0-flash-lite",
    },
]

# ─── Pricing (USD per 1K tokens) ─────────────────────────────
VERTEX_PRICING = {
    "gemini-2-5-pro":       {"input": 0.00125, "output": 0.010},
    "gemini-2-5-flash":     {"input": 0.00015, "output": 0.00060},
    "gemini-2-0-flash":     {"input": 0.00010, "output": 0.00040},
    "gemini-2-0-flash-lite":{"input": 0.000075,"output": 0.00030},
}


def _calculate_vertex_cost(short_id: str, input_tokens: int, output_tokens: int) -> float:
    pricing = VERTEX_PRICING.get(short_id, {"input": 0.0001, "output": 0.0004})
    return round(
        (input_tokens / 1000 * pricing["input"]) +
        (output_tokens / 1000 * pricing["output"]),
        6,
    )


def _call_single_vertex_model(model_id: str, provider: str, short_id: str, prompt: str) -> dict:
    """Call a single Vertex AI model synchronously (runs in thread pool)."""
    start = time.time()
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
    )
    latency_ms = int((time.time() - start) * 1000)

    text = response.text or ""

    # Token usage from usage_metadata
    usage = getattr(response, "usage_metadata", None)
    input_tokens  = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
    tokens = input_tokens + output_tokens
    cost = _calculate_vertex_cost(short_id, input_tokens, output_tokens)

    return {
        "model_id":   short_id,
        "provider":   provider,
        "response":   text,
        "tokens":     tokens,
        "cost":       cost,
        "latency_ms": latency_ms,
    }


async def call_all_vertex_models(prompt: str) -> list[dict]:
    """Call all Vertex AI models in parallel."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=len(VERTEX_MODELS)) as executor:
        futures = [
            loop.run_in_executor(
                executor,
                _call_single_vertex_model,
                m["model_id"],
                m["provider"],
                m["short_id"],
                prompt,
            )
            for m in VERTEX_MODELS
        ]
        results = await asyncio.gather(*futures, return_exceptions=True)

    clean = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"[VERTEX ERROR] {VERTEX_MODELS[i]['short_id']}: {r}")
        else:
            clean.append(r)
    return clean

"""
Round-robin Gemini evaluator client pool.

4 clients total:
  - 3 × Gemini API key clients  → gemini-2.5-flash  (free tier)
  - 1 × Vertex AI client        → gemini-2.5-flash  (paid Vertex quota)

Each client tracks which model it should use.
"""

import os
import threading
from google import genai

_CLIENTS: list[dict] = []   # [{"client": genai.Client, "model": str, "label": str}, ...]
_lock = threading.Lock()
_counter = 0
_initialized = False


def _init_clients():
    """Lazily initialize all evaluator clients from env vars."""
    global _CLIENTS, _initialized
    if _initialized:
        return

    clients = []

    # ── 3 Gemini API key clients → gemini-2.5-flash ──────────
    for var in ("GEMINI_API_KEY1", "GEMINI_API_KEY2", "GEMINI_API_KEY3"):
        k = os.getenv(var, "").strip()
        if k:
            clients.append({
                "client": genai.Client(api_key=k),
                "model":  "gemini-2.5-flash",
                "label":  f"{var} (flash)",
            })

    # ── 1 Vertex AI client → gemini-2.5-flash ──────────────────
    vertex_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if vertex_key:
        clients.append({
            "client": genai.Client(vertexai=True, api_key=vertex_key),
            "model":  "gemini-2.5-flash",
            "label":  "Vertex AI (flash)",
        })

    if not clients:
        raise RuntimeError("No Gemini/Vertex API keys found for evaluator pool")

    _CLIENTS = clients
    _initialized = True
    labels = [c["label"] for c in clients]
    print(f"[EVALUATOR POOL] Initialized {len(_CLIENTS)} clients: {labels}")


def get_client() -> dict:
    """
    Get the next evaluator client in round-robin order (thread-safe).
    Returns dict with: {"client": genai.Client, "model": str, "label": str}
    """
    global _counter
    _init_clients()

    with _lock:
        idx = _counter % len(_CLIENTS)
        _counter += 1

    return _CLIENTS[idx]


def get_client_count() -> int:
    """Return count of available evaluator clients."""
    _init_clients()
    return len(_CLIENTS)

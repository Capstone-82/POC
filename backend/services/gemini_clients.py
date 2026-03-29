"""
Round-robin evaluator client pool backed only by Vertex AI.

The pool uses known-working Vertex regional/global endpoints for
gemini-2.5-flash, with the global endpoint weighted more heavily.
"""

import os
import threading
import time

from google import genai

EVALUATOR_MODEL = "gemini-2.5-flash"
VERTEX_EVAL_REGIONS = [
    "asia-south1",
    "asia-southeast1",
    "asia-northeast1",
    "asia-northeast3",
    "australia-southeast1",
    "us-central1",
    "us-east4",
    "us-east1",
    "us-west1",
    "us-west4",
    "northamerica-northeast1",
    "europe-west1",
    "europe-west2",
    "europe-west3",
    "europe-west4",
    "europe-west9",
    "southamerica-east1",
    "global",
]

_CLIENTS: list[dict] = []
_lock = threading.Lock()
_counter = 0
_initialized = False
_cooldowns: dict[str, float] = {}


def _vertex_weight(location: str) -> int:
    if location == "global":
        return int(os.getenv("VERTEX_EVAL_GLOBAL_WEIGHT", "6"))
    return int(os.getenv("VERTEX_EVAL_REGION_WEIGHT", "1"))


def _build_vertex_entry(location: str) -> dict:
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for Vertex evaluator clients")

    client = genai.Client(
        vertexai=True,
        project=project,
        location=location,
    )
    return {
        "client": client,
        "model": EVALUATOR_MODEL,
        "label": f"Vertex AI ({location})",
        "source": "vertex",
        "location": location,
    }


def _init_clients():
    """Lazily initialize all evaluator clients from env vars."""
    global _CLIENTS, _initialized
    if _initialized:
        return

    clients = []

    vertex_project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    if vertex_project:
        for location in VERTEX_EVAL_REGIONS:
            entry = _build_vertex_entry(location)
            weight = max(1, _vertex_weight(location))
            for replica_index in range(weight):
                clients.append({
                    **entry,
                    "label": (
                        f"{entry['label']} [weight {replica_index + 1}/{weight}]"
                        if weight > 1 else entry["label"]
                    ),
                })

    if not clients:
        raise RuntimeError("No Vertex evaluator clients could be initialized")

    _CLIENTS = clients
    _initialized = True
    labels = [client["label"] for client in clients]
    print(f"[EVALUATOR POOL] Initialized {len(_CLIENTS)} clients: {labels}")


def get_client() -> dict:
    """
    Get the next evaluator client in round-robin order (thread-safe).
    Returns dict with client metadata.
    """
    global _counter
    _init_clients()

    with _lock:
        now = time.time()
        available = [
            client for client in _CLIENTS
            if _cooldowns.get(client["label"], 0) <= now
        ]
        if not available:
            available = _CLIENTS

        idx = _counter % len(available)
        _counter += 1

    return available[idx]


def get_client_count() -> int:
    _init_clients()
    return len(_CLIENTS)


def cooldown_client(label: str, seconds: float):
    _init_clients()
    with _lock:
        _cooldowns[label] = time.time() + max(0.0, seconds)

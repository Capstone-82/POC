"""
Smoke-test Gemini model access using multiple Google AI Studio API keys.

This script:
- loads backend/.env
- reads GEMINI_API_KEY1..GEMINI_API_KEY4 (and GOOGLE_API_KEY as fallback)
- uses round-robin key selection
- retries on rate-limit style errors by switching to the next key
- exercises the Gemini model ids currently referenced by the repo

Usage:
  python backend/test_gemini_models.py
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai


def _load_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / "backend" / ".env")


_load_env()

TEST_PROMPT = "Reply with exactly: GEMINI_OK"
MAX_RETRIES_PER_MODEL = 4

# These are the Gemini model ids currently referenced by backend/services/vertex.py.
GEMINI_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


class GeminiRoundRobinPool:
    def __init__(self, api_keys: list[str]):
        if not api_keys:
            raise RuntimeError("No Gemini API keys found in backend/.env")
        self._clients = [
            {
                "label": f"key{index + 1}",
                "client": genai.Client(api_key=api_key),
            }
            for index, api_key in enumerate(api_keys)
        ]
        self._counter = 0
        self._cooldowns: dict[str, float] = {}
        self._lock = threading.Lock()

    def next_client(self) -> dict:
        with self._lock:
            now = time.time()
            available = [
                entry for entry in self._clients
                if self._cooldowns.get(entry["label"], 0) <= now
            ]
            if not available:
                available = self._clients

            entry = available[self._counter % len(available)]
            self._counter += 1
            return entry

    def cooldown(self, label: str, seconds: float) -> None:
        with self._lock:
            self._cooldowns[label] = time.time() + max(0.0, seconds)

    @property
    def count(self) -> int:
        return len(self._clients)


def _collect_api_keys() -> list[str]:
    import os

    keys: list[str] = []
    for env_name in ("GEMINI_API_KEY1", "GEMINI_API_KEY2", "GEMINI_API_KEY3", "GEMINI_API_KEY4", "GOOGLE_API_KEY","GEMINI_API_KEY5","GEMINI_API_KEY6","GEMINI_API_KEY7",):
        value = os.getenv(env_name, "").strip()
        if value and value not in keys:
            keys.append(value)
    return keys


def _is_retryable(exc: Exception) -> bool:
    message = str(exc).lower()
    retry_markers = [
        "429",
        "rate",
        "quota",
        "resource exhausted",
        "unavailable",
        "deadline",
        "timeout",
        "503",
        "overloaded",
    ]
    return any(marker in message for marker in retry_markers)


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def test_model(pool: GeminiRoundRobinPool, model_id: str) -> bool:
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES_PER_MODEL):
        entry = pool.next_client()
        client = entry["client"]
        label = entry["label"]
        try:
            started = time.time()
            response = client.models.generate_content(
                model=model_id,
                contents=TEST_PROMPT,
            )
            latency_ms = int((time.time() - started) * 1000)
            text = _normalize_text(response.text or "")
            print(
                f"PASS  {model_id:<28} via={label:<5} latency={latency_ms:>5}ms "
                f"text={text[:80]!r}"
            )
            return True
        except Exception as exc:
            last_error = exc
            if _is_retryable(exc):
                pool.cooldown(label, seconds=15.0)
                print(
                    f"RETRY {model_id:<28} via={label:<5} attempt={attempt + 1}/{MAX_RETRIES_PER_MODEL} "
                    f"error={exc}"
                )
                continue

            print(f"FAIL  {model_id:<28} via={label:<5} error={exc}")
            return False

    print(f"FAIL  {model_id:<28} error={last_error}")
    return False


def main() -> int:
    api_keys = _collect_api_keys()
    pool = GeminiRoundRobinPool(api_keys)

    print("=" * 100)
    print("GEMINI MODEL ACCESS TEST (ROUND ROBIN ACROSS AI STUDIO KEYS)")
    print("=" * 100)
    print(f"Keys loaded: {pool.count}")
    print(f"Models configured: {len(GEMINI_MODELS)}")
    print("-" * 100)

    passed = 0
    for model_id in GEMINI_MODELS:
        if test_model(pool, model_id):
            passed += 1

    print("-" * 100)
    print(f"Result: {passed}/{len(GEMINI_MODELS)} Gemini models succeeded")
    print("=" * 100)
    return 0 if passed == len(GEMINI_MODELS) else 1


if __name__ == "__main__":
    raise SystemExit(main())

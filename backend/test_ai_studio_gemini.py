"""
Test all Gemini model ids referenced by this repo using a normal Google AI Studio API key.

Usage:
  python backend/test_ai_studio_gemini.py

Reads:
  - GOOGLE_API_KEY
  - GEMINI_TEST_PROMPT (optional)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai


def _load_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / "backend" / ".env")


_load_env()

TEST_PROMPT = os.getenv("GEMINI_TEST_PROMPT", "Reply with exactly: OK").strip() or "Reply with exactly: OK"
API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

# Benchmark-time Gemini models from backend/services/vertex.py
BENCHMARK_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

# Legacy evaluator pool model from backend/services/gemini_clients.py
EVALUATOR_MODELS = [
    "gemini-2.5-flash",
]


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


ALL_MODELS = _unique(BENCHMARK_MODELS + EVALUATOR_MODELS)


def build_client() -> genai.Client:
    if not API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is missing. This script expects a normal Google AI Studio API key.")
    return genai.Client(api_key=API_KEY)


def test_model(client: genai.Client, model_id: str) -> bool:
    try:
        started = time.time()
        response = client.models.generate_content(
            model=model_id,
            contents=TEST_PROMPT,
        )
        latency_ms = int((time.time() - started) * 1000)
        text = (response.text or "").strip().replace("\n", " ")
        print(f"PASS  {model_id:<30} -> {text[:80]!r} ({latency_ms}ms)")
        return True
    except Exception as exc:
        print(f"FAIL  {model_id:<30} -> {exc}")
        return False


def main() -> int:
    print("=" * 88)
    print("GOOGLE AI STUDIO GEMINI MODEL ACCESS TEST")
    print("=" * 88)
    print(f"API key present: {'yes' if API_KEY else 'no'}")
    print(f"Prompt: {TEST_PROMPT}")
    print("-" * 88)

    client = build_client()
    passed = 0
    for model_id in ALL_MODELS:
        if test_model(client, model_id):
            passed += 1

    print("-" * 88)
    print(f"Result: {passed}/{len(ALL_MODELS)} model ids succeeded")
    print("=" * 88)
    return 0 if passed == len(ALL_MODELS) else 1


if __name__ == "__main__":
    raise SystemExit(main())

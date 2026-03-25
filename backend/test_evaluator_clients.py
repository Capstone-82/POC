"""
Test script — verifies all 4 evaluator clients.
  - 3 Gemini API keys evaluate using gemini-2.5-flash (free tier)
  - 1 Vertex AI key evaluates using gemini-2.5-flash (paid/Vertex)
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

from google import genai

TEST_PROMPT = "What is 2 + 2? Reply with just the number."


def test_gemini_api_key(label: str, api_key: str):
    """Test a Gemini API key client specifically with gemini-2.5-flash."""
    try:
        model = "gemini-2.5-flash"
        client = genai.Client(api_key=api_key)
        start = time.time()
        response = client.models.generate_content(model=model, contents=TEST_PROMPT)
        latency = int((time.time() - start) * 1000)
        text = (response.text or "").strip()[:100]
        print(f"  ✅ [gemini-2.5-flash] {label}: '{text}' ({latency}ms)")
        return True
    except Exception as e:
        print(f"  ❌ {label}: {e}")
        return False


def test_vertex_client(label: str, api_key: str):
    """Test a Vertex AI client specifically with gemini-2.5-flash."""
    try:
        model = "gemini-2.5-flash"
        client = genai.Client(vertexai=True, api_key=api_key)
        start = time.time()
        response = client.models.generate_content(model=model, contents=TEST_PROMPT)
        latency = int((time.time() - start) * 1000)
        text = (response.text or "").strip()[:100]
        print(f"  ✅ [gemini-2.5-flash] {label}: '{text}' ({latency}ms)")
        return True
    except Exception as e:
        print(f"  ❌ {label}: {e}")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("EVALUATOR CLIENT POOL TEST (Multi-Model)")
    print("=" * 70)

    passed = 0
    total = 0

    # Test 3 Gemini API keys (Free tier -> Flash)
    print("\n-- Gemini API Key Clients (Free Tier / Flash) --")
    for i, var in enumerate(["GEMINI_API_KEY1", "GEMINI_API_KEY2", "GEMINI_API_KEY3"], 1):
        key = os.getenv(var, "").strip()
        total += 1
        if not key:
            print(f"  ⚠️  {var}: NOT SET in .env")
        else:
            if test_gemini_api_key(f"Client {i} ({var})", key):
                passed += 1

    # Test Vertex AI client (Paid -> Flash)
    print("\n-- Vertex AI Client (Paid / Flash) --")
    vertex_key = os.getenv("GOOGLE_API_KEY", "").strip()
    total += 1
    if not vertex_key:
        print("  ⚠️  GOOGLE_API_KEY: NOT SET in .env")
    else:
        if test_vertex_client("Client 4 (Vertex AI)", vertex_key):
            passed += 1

    print("\n" + "=" * 70)
    print(f"RESULT: {passed}/{total} clients working")
    if passed == total:
        print("🎉 All 4 evaluator clients are operational!")
    else:
        print(f"⚠️  {total - passed} client(s) failed — check API keys above")
    print("=" * 70)

"""
Test script for the Vertex-only evaluator pool.

It checks the global endpoint plus all configured evaluator regions using
service-account / ADC auth.
"""

import time

from dotenv import load_dotenv
from google import genai

from services.gemini_clients import EVALUATOR_MODEL, VERTEX_EVAL_REGIONS

load_dotenv()

TEST_PROMPT = "What is 2 + 2? Reply with just the number."


def test_vertex_location(project: str, location: str):
    try:
        client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )
        start = time.time()
        response = client.models.generate_content(model=EVALUATOR_MODEL, contents=TEST_PROMPT)
        latency = int((time.time() - start) * 1000)
        text = (response.text or "").strip()[:100]
        print(f"  PASS  [{EVALUATOR_MODEL}] {location}: '{text}' ({latency}ms)")
        return True
    except Exception as exc:
        print(f"  FAIL  {location}: {exc}")
        return False


if __name__ == "__main__":
    import os

    print("=" * 78)
    print("VERTEX-ONLY EVALUATOR POOL TEST")
    print("=" * 78)

    project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is not set in .env")

    passed = 0
    total = len(VERTEX_EVAL_REGIONS)
    for location in VERTEX_EVAL_REGIONS:
        if test_vertex_location(project, location):
            passed += 1

    print("-" * 78)
    print(f"RESULT: {passed}/{total} Vertex evaluator locations succeeded")
    print("=" * 78)

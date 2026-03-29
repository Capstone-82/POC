"""
Test Vertex AI Gemini access across multiple locations/endpoints.

Usage:
  python test_vertex_regions.py

Reads from .env:
  - GOOGLE_CLOUD_PROJECT
  - GOOGLE_CLOUD_LOCATION (optional, only shown for reference)
  - GOOGLE_APPLICATION_CREDENTIALS (optional, can also be set in shell)
"""

import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()

TEST_PROMPT = "What is 2 + 2? Reply with just the number."
TEST_MODEL = os.getenv("VERTEX_TEST_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"

PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
DEFAULT_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "").strip()
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()

LOCATIONS = [
    "asia-south1",
    "asia-east1",
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
    "northamerica-northeast2",
    "europe-west1",
    "europe-west2",
    "europe-west3",
    "europe-west4",
    "europe-west6",
    "europe-west9",
    "southamerica-east1",
    "global",
]


def build_client(location: str):
    return genai.Client(
        vertexai=True,
        project=PROJECT,
        location=location,
    )


def test_location(location: str) -> bool:
    try:
        client = build_client(location)
        start = time.time()
        response = client.models.generate_content(
            model=TEST_MODEL,
            contents=TEST_PROMPT,
        )
        latency_ms = int((time.time() - start) * 1000)
        text = (response.text or "").strip()
        print(f"  SUCCESS  {location:<24} -> '{text}' ({latency_ms}ms)")
        return True
    except Exception as exc:
        print(f"  FAIL     {location:<24} -> {exc}")
        return False


if __name__ == "__main__":
    print("=" * 78)
    print("VERTEX REGION / GLOBAL ENDPOINT TEST")
    print("=" * 78)
    print(f"Model:              {TEST_MODEL}")
    print(f"Project:            {PROJECT or '[missing GOOGLE_CLOUD_PROJECT]'}")
    print(f"Default env region: {DEFAULT_LOCATION or '[missing GOOGLE_CLOUD_LOCATION]'}")
    print(
        "Credentials file:   "
        f"{GOOGLE_APPLICATION_CREDENTIALS or '[using externally provided ADC or missing GOOGLE_APPLICATION_CREDENTIALS]'}"
    )
    print("-" * 78)

    if not PROJECT:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is missing from .env")

    passed = 0
    for location in LOCATIONS:
        if test_location(location):
            passed += 1

    print("-" * 78)
    print(f"Result: {passed}/{len(LOCATIONS)} locations succeeded")
    print("=" * 78)

"""
Smoke-test every configured Bedrock model using credentials from backend/.env.

Usage:
  python backend/test_bedrock_models.py

Reads:
  - AWS_ACCESS_KEY_ID
  - AWS_SECRET_ACCESS_KEY
  - AWS_REGION
  - AWS_ACCOUNT_ID
  - BEDROCK_TEST_PROMPT (optional)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


def _load_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / "backend" / ".env")
    sys.path.insert(0, str(repo_root / "backend"))


_load_env()

from services.bedrock import BEDROCK_MODELS, _build_body, _extract_text, _extract_tokens, bedrock  # noqa: E402


TEST_PROMPT = "Reply with exactly: BEDROCK_OK"


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def test_model(model: dict) -> bool:
    try:
        started = time.time()
        raw = bedrock.invoke_model(
            modelId=model["model_id"],
            body=_build_body(model["fmt"], TEST_PROMPT),
            contentType="application/json",
            accept="application/json",
        )
        latency_ms = int((time.time() - started) * 1000)
        body_json = json.loads(raw["body"].read())
        text = _normalize_text(_extract_text(model["fmt"], body_json))
        input_tokens, output_tokens = _extract_tokens(model["fmt"], body_json, TEST_PROMPT)
        print(
            f"PASS  {model['short_id']:<18} provider={model['provider']:<10} "
            f"tokens={input_tokens + output_tokens:<5} latency={latency_ms:>5}ms "
            f"text={text[:80]!r}"
        )
        return True
    except Exception as exc:
        print(
            f"FAIL  {model['short_id']:<18} provider={model['provider']:<10} "
            f"error={exc}"
        )
        return False


def main() -> int:
    print("=" * 100)
    print("BEDROCK MODEL ACCESS TEST")
    print("=" * 100)
    print(f"Models configured: {len(BEDROCK_MODELS)}")
    print("-" * 100)

    passed = 0
    for model in BEDROCK_MODELS:
        if test_model(model):
            passed += 1

    print("-" * 100)
    print(f"Result: {passed}/{len(BEDROCK_MODELS)} Bedrock models succeeded")
    print("=" * 100)
    return 0 if passed == len(BEDROCK_MODELS) else 1


if __name__ == "__main__":
    raise SystemExit(main())

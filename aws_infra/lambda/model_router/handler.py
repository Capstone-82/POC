"""
Lambda handler — Model Router.

Flow:
  1. Parse app_id, env, model_id, prompt, max_output_tokens from the event.
  2. Load the app config from SSM (cached).
  3. Check the allow-list.
  4. Check the context window limits.
  5. Forward the approved request to Bedrock OR return a structured rejection.

Expected event shape (from API Gateway proxy integration):
{
  "app_id":           "modelmatrix",
  "env":              "prod",           // optional, defaults to "prod"
  "model_id":         "nova-lite",
  "prompt":           "Explain quantum entanglement in simple terms.",
  "max_output_tokens": 512             // optional, 0 = use app default
}
"""

import json
import os
import boto3
from config_loader import get_app_config
from token_estimator import check_context_limits

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

_bedrock = None


def _get_bedrock():
    global _bedrock
    if _bedrock is None:
        _bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _bedrock


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _invoke_bedrock(model_id: str, prompt: str, max_output_tokens: int) -> str:
    """Invoke a Bedrock model using the Converse API (model-agnostic)."""
    client = _get_bedrock()
    resp = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_output_tokens or 1024},
    )
    return resp["output"]["message"]["content"][0]["text"]


def handler(event, context):
    # ── 1. Parse event ────────────────────────────────────────
    if isinstance(event.get("body"), str):
        body = json.loads(event["body"])
    else:
        body = event.get("body", event)

    app_id           = body.get("app_id", "")
    env              = body.get("env", "prod")
    model_id         = body.get("model_id", "")
    prompt           = body.get("prompt", "")
    max_output_tokens = int(body.get("max_output_tokens", 0))

    if not app_id or not model_id or not prompt:
        return _response(400, {
            "error": "bad_request",
            "message": "app_id, model_id, and prompt are required.",
        })

    # ── 2. Load config ────────────────────────────────────────
    config = get_app_config(app_id, env)
    if config is None:
        return _response(404, {
            "error": "app_not_found",
            "message": f"No configuration found for app '{app_id}' / env '{env}'.",
        })

    # ── 3. Allow-List check ───────────────────────────────────
    allowed_models = [m.lower() for m in config.get("allowed_models", [])]
    if model_id.lower() not in allowed_models:
        return _response(403, {
            "error": "model_not_allowed",
            "message": (
                f"Model '{model_id}' is not on the allow-list for "
                f"'{app_id}/{env}'. Allowed: {config.get('allowed_models', [])}."
            ),
            "allowed_models": config.get("allowed_models", []),
        })

    # ── 4. Context Window check ───────────────────────────────
    limits  = config.get("context_limits", {})
    ctx_chk = check_context_limits(prompt, max_output_tokens, limits)
    if not ctx_chk["allowed"]:
        return _response(400, {
            "error": "context_window_exceeded",
            "message": ctx_chk["message"],
            "violation": ctx_chk["violation"],
            "estimated_input_tokens":  ctx_chk["estimated_input_tokens"],
            "estimated_output_tokens": ctx_chk["estimated_output_tokens"],
            "estimated_total_tokens":  ctx_chk["estimated_total_tokens"],
            "limits": limits,
        })

    # ── 5. Forward to Bedrock ─────────────────────────────────
    try:
        text = _invoke_bedrock(model_id, prompt, max_output_tokens)
        return _response(200, {
            "allowed": True,
            "model_id": model_id,
            "app_id": app_id,
            "env": env,
            "response": text,
            "estimated_input_tokens":  ctx_chk["estimated_input_tokens"],
            "estimated_output_tokens": ctx_chk["estimated_output_tokens"],
            "estimated_total_tokens":  ctx_chk["estimated_total_tokens"],
        })
    except Exception as exc:
        return _response(502, {
            "error": "model_invocation_failed",
            "message": str(exc),
        })

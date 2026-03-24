"""
Test router — Bedrock + Vertex AI model testing.
GET  /api/test/models            → list all configured models (Bedrock + Vertex)
POST /api/test/model/{short_id}  → call one model directly
POST /api/test/all               → call every model, return pass/fail summary
POST /api/test/vertex            → call only Vertex AI (Gemini) models
"""

import time
from fastapi import APIRouter
from services.bedrock import BEDROCK_MODELS, _call_single_model
from services.vertex import VERTEX_MODELS, _call_single_vertex_model

router = APIRouter()


# ─── List all configured models ──────────────────────────────

@router.get("/available-ids")
async def list_available_ids(provider: str = None):
    """List all model IDs directly from AWS Bedrock for debugging."""
    try:
        params = {}
        if provider:
            params["byProvider"] = provider
        
        response = bedrock.list_foundation_models(**params)
        ids = [m["modelId"] for m in response["modelSummaries"]]
        return {"available_ids": ids}
    except Exception as e:
        return {"error": str(e)}


@router.get("/models")
def list_models():
    bedrock_list = [
        {"source": "bedrock", "short_id": m["short_id"], "provider": m["provider"], "model_id": m["model_id"]}
        for m in BEDROCK_MODELS
    ]
    vertex_list = [
        {"source": "vertex", "short_id": m["short_id"], "provider": m["provider"], "model_id": m["model_id"]}
        for m in VERTEX_MODELS
    ]
    all_models = bedrock_list + vertex_list
    return {"total": len(all_models), "models": all_models}


# ─── Test a single model by short_id ─────────────────────────

@router.post("/model/{short_id}")
def test_single(short_id: str, body: dict = None):
    prompt = (body or {}).get("prompt", "Say hello in one sentence.")

    # Check Bedrock first
    bmodel = next((m for m in BEDROCK_MODELS if m["short_id"] == short_id), None)
    if bmodel:
        try:
            r = _call_single_model(bmodel["model_id"], bmodel["provider"], bmodel["short_id"], bmodel["fmt"], prompt)
            return {"status": "ok", "source": "bedrock", **r}
        except Exception as e:
            return {"status": "error", "source": "bedrock", "short_id": short_id, "error": str(e)}

    # Check Vertex
    vmodel = next((m for m in VERTEX_MODELS if m["short_id"] == short_id), None)
    if vmodel:
        try:
            r = _call_single_vertex_model(vmodel["model_id"], vmodel["provider"], vmodel["short_id"], prompt)
            return {"status": "ok", "source": "vertex", **r}
        except Exception as e:
            return {"status": "error", "source": "vertex", "short_id": short_id, "error": str(e)}

    return {"error": f"Unknown short_id: {short_id}. See GET /api/test/models"}


# ─── Test ALL Bedrock models ──────────────────────────────────

@router.post("/all")
def test_all():
    prompt = "Say hello in exactly one sentence."
    provider_order = ["Meta", "Amazon", "Mistral AI", "DeepSeek", "Anthropic"]
    ordered = sorted(
        BEDROCK_MODELS,
        key=lambda m: provider_order.index(m["provider"]) if m["provider"] in provider_order else 99,
    )

    results = []
    for model in ordered:
        start = time.time()
        try:
            r = _call_single_model(model["model_id"], model["provider"], model["short_id"], model["fmt"], prompt)
            results.append({
                "status":           "✅ PASS",
                "source":           "bedrock",
                "provider":         model["provider"],
                "short_id":         model["short_id"],
                "response_preview": (r["response"] or "")[:120],
                "tokens":           r["tokens"],
                "latency_ms":       r["latency_ms"],
                "cost_usd":         r["cost"],
            })
        except Exception as e:
            results.append({
                "status":     "❌ FAIL",
                "source":     "bedrock",
                "provider":   model["provider"],
                "short_id":   model["short_id"],
                "error":      str(e),
                "latency_ms": int((time.time() - start) * 1000),
            })

    passed = sum(1 for r in results if "PASS" in r["status"])
    return {"summary": f"{passed}/{len(results)} Bedrock models passed", "results": results}


# ─── Test ONLY Vertex AI (Gemini) models ─────────────────────

@router.post("/vertex")
def test_vertex():
    prompt = "Say hello in exactly one sentence."
    results = []

    for model in VERTEX_MODELS:
        start = time.time()
        try:
            r = _call_single_vertex_model(model["model_id"], model["provider"], model["short_id"], prompt)
            results.append({
                "status":           "✅ PASS",
                "source":           "vertex",
                "provider":         model["provider"],
                "short_id":         model["short_id"],
                "response_preview": (r["response"] or "")[:120],
                "tokens":           r["tokens"],
                "latency_ms":       r["latency_ms"],
                "cost_usd":         r["cost"],
            })
        except Exception as e:
            results.append({
                "status":     "❌ FAIL",
                "source":     "vertex",
                "provider":   model["provider"],
                "short_id":   model["short_id"],
                "error":      str(e),
                "latency_ms": int((time.time() - start) * 1000),
            })

    passed = sum(1 for r in results if "PASS" in r["status"])
    return {"summary": f"{passed}/{len(results)} Vertex AI models passed", "results": results}

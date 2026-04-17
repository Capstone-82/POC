import json
import re
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from models.schemas import InferenceRequest, InferenceResponse
from services.recommender import get_recommendation, get_recommendation_options

router = APIRouter()

# ─── Gemini short_ids — excluded from A/B test (billing disabled) ─────────────
GEMINI_SHORT_IDS = {
    "gemini-2-0-flash", "gemini-2-0-flash-lite",
    "gemini-2-5-flash", "gemini-2-5-pro",
    "gemini-3-1-pro",   "gemini-3-1-flash-lite",
}

# ─── Evaluator models (all Bedrock — no Vertex dependency) ───────────────────
EVALUATOR_SHORT_IDS = {"llama4-maverick", "mistral-large", "nova-premier"}

# Evaluator prompt (matches generate_avg_accuracy_scores.py rubric)
_EVALUATOR_SYSTEM = """\
You are an expert evaluator for LLM benchmark datasets.
You will receive: the use case, the original user prompt, and one model response.
Score how well the response satisfies the prompt for that use case.

Score meaning:
  95-100 excellent  |  85-94 strong  |  75-84 good  |  65-74 mixed
  50-64  weak       |  30-49 poor    |  10-29 very poor  |  0-9 no value

Return ONLY valid JSON in exactly this shape:
{"accuracy_score": <integer 0-100>}
"""

_USE_CASE_RUBRICS = {
    "code-generation": (
        "Evaluate for: correctness of logic/syntax (50%), "
        "completeness (25%), code clarity and best practices (25%). "
        "Deduct heavily for broken or non-runnable code."
    ),
    "reasoning": (
        "Evaluate for: logical correctness (50%), "
        "completeness of reasoning chain (30%), clarity (20%). "
        "Deduct for logical gaps or unsupported conclusions."
    ),
    "text-generation": (
        "Evaluate for: correctness and relevance (55%), "
        "scope fit (25%), completeness (15%), clarity (5%). "
        "Do not reward unnecessary length."
    ),
}


def _build_eval_prompt(use_case: str, prompt: str, response: str) -> str:
    rubric = _USE_CASE_RUBRICS.get(use_case, _USE_CASE_RUBRICS["text-generation"])
    return (
        f"{_EVALUATOR_SYSTEM}\n"
        f"USE CASE: {use_case}\n"
        f"RUBRIC: {rubric}\n\n"
        f"PROMPT:\n{prompt}\n\n"
        f"RESPONSE TO EVALUATE:\n{response}\n\n"
        f"Return ONLY JSON: {{\"accuracy_score\": <0-100>}}"
    )


def _parse_score(text: str) -> Optional[float]:
    """Extract accuracy_score from evaluator JSON output."""
    # Try full JSON parse first
    try:
        m = re.search(r'\{[^{}]*"accuracy_score"\s*:\s*(\d+)[^{}]*\}', text, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            return float(parsed["accuracy_score"])
    except Exception:
        pass
    # Fallback: bare number extraction
    m2 = re.search(r'"accuracy_score"\s*:\s*(\d+)', text)
    if m2:
        return float(m2.group(1))
    return None


# ─── Schemas ─────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    prompt: str
    model: str
    use_case: Optional[str] = "text-generation"

class RunResponse(BaseModel):
    model_used: str
    provider: str
    response: str
    latency_ms: int
    cost: float
    tokens: int

class EvaluateRequest(BaseModel):
    prompt: str
    response: str
    use_case: str
    evaluator_model: str

class EvaluateResponse(BaseModel):
    evaluator_model: str
    score: Optional[float] = None
    error: Optional[str]   = None


# ─── Existing endpoints ───────────────────────────────────────────────────────

@router.get("/options")
async def recommendation_options():
    return await get_recommendation_options()


@router.post("/recommend", response_model=InferenceResponse)
async def recommend(req: InferenceRequest):
    result = await get_recommendation(
        use_case=req.use_case.value if hasattr(req.use_case, "value") else req.use_case,
        prompt=req.prompt,
        current_model=req.current_model,
        min_accuracy_gain=req.min_accuracy_gain,
        max_cost_increase_pct=req.max_cost_increase_pct,
        max_latency_increase_pct=req.max_latency_increase_pct,
    )
    return InferenceResponse(**result)


# ─── POST /run ────────────────────────────────────────────────────────────────

@router.post("/run", response_model=RunResponse)
async def run_model(req: RunRequest):
    """
    Call a Bedrock model by short_id. Gemini/Vertex models are excluded
    (billing disabled on Vertex AI project).
    """
    import asyncio
    from services.bedrock import (
        BEDROCK_MODELS, _build_body, _extract_text,
        _extract_tokens, bedrock as bedrock_client, MODEL_PRICING,
    )

    short_id = req.model.strip()

    # Hard-block Gemini — Vertex billing is disabled
    if short_id in GEMINI_SHORT_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Gemini models are excluded from A/B testing (Vertex billing disabled). "
                   f"Model requested: '{short_id}'",
        )

    bk_match = next((m for m in BEDROCK_MODELS if m["short_id"] == short_id), None)
    if not bk_match:
        raise HTTPException(status_code=404, detail=f"Unknown model short_id: '{short_id}'")

    loop = asyncio.get_event_loop()

    def _call():
        body      = _build_body(bk_match["fmt"], req.prompt)
        t0        = time.time()
        raw       = bedrock_client.invoke_model(
            modelId=bk_match["model_id"], body=body,
            contentType="application/json", accept="application/json",
        )
        lat_ms    = int((time.time() - t0) * 1000)
        body_json = json.loads(raw["body"].read())
        text      = _extract_text(bk_match["fmt"], body_json)
        inp, out  = _extract_tokens(bk_match["fmt"], body_json, req.prompt)
        pricing   = MODEL_PRICING.get(short_id, {"input": 0.001, "output": 0.003})
        cost      = round((inp / 1000 * pricing["input"]) + (out / 1000 * pricing["output"]), 6)
        return text, inp + out, cost, lat_ms

    text, tokens, cost, latency_ms = await loop.run_in_executor(None, _call)
    return RunResponse(
        model_used=short_id,
        provider=bk_match["provider"],
        response=text,
        latency_ms=latency_ms,
        cost=cost,
        tokens=tokens,
    )


# ─── POST /evaluate ──────────────────────────────────────────────────────────

@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_response(req: EvaluateRequest):
    """
    Score a (prompt, response) pair using a Bedrock evaluator model.
    Supported: llama4-maverick, mistral-large, nova-premier.
    Returns 0-100 accuracy_score.
    """
    import asyncio
    from services.bedrock import (
        BEDROCK_MODELS, _build_body, _extract_text, bedrock as bedrock_client,
    )

    short_id = req.evaluator_model.strip()

    if short_id not in EVALUATOR_SHORT_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported evaluator '{short_id}'. "
                   f"Choose from: {sorted(EVALUATOR_SHORT_IDS)}",
        )

    bk_match = next((m for m in BEDROCK_MODELS if m["short_id"] == short_id), None)
    if not bk_match:
        raise HTTPException(status_code=404, detail=f"Evaluator '{short_id}' not in Bedrock registry")

    eval_prompt = _build_eval_prompt(req.use_case, req.prompt, req.response)
    loop        = asyncio.get_event_loop()

    def _run():
        try:
            body      = _build_body(bk_match["fmt"], eval_prompt)
            raw       = bedrock_client.invoke_model(
                modelId=bk_match["model_id"], body=body,
                contentType="application/json", accept="application/json",
            )
            body_json = json.loads(raw["body"].read())
            text      = _extract_text(bk_match["fmt"], body_json)
            score     = _parse_score(text)
            if score is None:
                return None, f"Could not parse score from: {text[:300]}"
            return round(min(max(score, 0.0), 100.0), 2), None
        except Exception as exc:
            return None, str(exc)

    score, error = await loop.run_in_executor(None, _run)
    return EvaluateResponse(evaluator_model=short_id, score=score, error=error)

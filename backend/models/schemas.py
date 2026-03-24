from pydantic import BaseModel
from typing import Optional


# ─── Training ────────────────────────────────────────────────

class SinglePromptRequest(BaseModel):
    prompt: str
    prompt_complexity: str = "mid"          # low | mid | high — chosen by user in frontend
    prompt_quality_score: int = 50          # user-provided accuracy from frontend (0-100)
    evaluator_model: str = "gemini-2.0-flash"


class JobResponse(BaseModel):
    job_id: str


# SSE event shapes (serialized to JSON strings in the stream)
class LogEvent(BaseModel):
    type: str                # "progress" | "done" | "error"
    prompt_index: int
    total: int
    model_id: str
    provider: str
    prompt_complexity: str
    prompt_quality_score: int
    accuracy_score: int
    cost: float
    tokens: int
    latency_ms: int


# ─── Inference ───────────────────────────────────────────────

class InferenceRequest(BaseModel):
    prompt: str
    use_case: str
    current_model: str


class InferenceResponse(BaseModel):
    complexity: str
    quality_score: int
    current_model: str
    recommended_model: str
    accuracy_delta: float
    cost_delta: float
    latency_delta: int
    reason: str

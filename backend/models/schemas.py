from pydantic import BaseModel
from typing import Optional
from enum import Enum


# ─── Enums ───────────────────────────────────────────────────

class ClarityLevel(str, Enum):
    CLEAR = "CLEAR"
    PARTIAL = "PARTIAL"
    UNCLEAR = "UNCLEAR"


class UseCase(str, Enum):
    TEXT_GENERATION = "text-generation"
    REASONING = "reasoning"
    CODE_GENERATION = "code-generation"


class PromptComplexity(str, Enum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"


# ─── Training ────────────────────────────────────────────────

class SinglePromptRequest(BaseModel):
    prompt: str
    prompt_complexity: PromptComplexity = PromptComplexity.MID
    use_case: UseCase = UseCase.TEXT_GENERATION
    clarity: ClarityLevel = ClarityLevel.CLEAR


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
    use_case: str
    clarity: str
    accuracy_score: int
    cost: float
    tokens: int
    latency_ms: int


# ─── Inference ───────────────────────────────────────────────

class InferenceRequest(BaseModel):
    prompt: str
    use_case: UseCase
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

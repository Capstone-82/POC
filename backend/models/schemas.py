from pydantic import BaseModel
from typing import List, Optional
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


class ModelStats(BaseModel):
    model_id: str
    provider: str
    sample_count: int
    avg_accuracy: float
    median_accuracy: float
    median_cost: float
    median_latency_ms: float


class InferenceResponse(BaseModel):
    complexity: str
    complexity_confidence: Optional[float] = None
    complexity_source: str
    quality_score: Optional[int] = None
    clarity: str
    clarity_source: str
    filter_level: str
    recommendation_mode: str
    data_source: str
    current_model: str
    current_model_found: bool
    current_model_stats: Optional[ModelStats] = None
    recommended_model: str
    recommended_provider: str
    expected_accuracy: float
    expected_cost: float
    expected_latency: float
    accuracy_delta: Optional[float] = None
    accuracy_delta_pct: Optional[float] = None
    cost_delta_pct: Optional[float] = None
    latency_delta_pct: Optional[float] = None
    sample_size: int
    slice_row_count: int
    models_considered: int
    switch_recommended: bool
    final_suggestion_model: str
    policy_reason: str
    reason: str
    top_candidates: List[ModelStats]
    warnings: List[str] = []

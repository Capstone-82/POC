from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class PromptProfile:
    d1: float
    d2: float
    d3: float
    d4: float
    d5: float
    domain: str
    complexity_score: float
    derived_tier: str               # T1/T2/T3
    intent: str                     # FACTUAL/ANALYTICAL/SYNTHETIC/STRATEGIC
    task_type: str
    reasoning_chain_detected: bool
    research_signals: List[str]
    confidence: float
    input_token_count: int          # Real count from prompt text
    est_output_tokens: int          # d3-bucket or caller override

@dataclass
class ModelCandidate:
    model_id: str
    provider: str
    generation: str                 # current/previous/legacy
    tier: str
    cost_in: float                  # per 1M input tokens
    cost_out: float                 # per 1M output tokens
    max_input_tokens: int
    max_output_tokens: int
    tool_tier: int
    reasoning_mode: bool
    speed_tokens_per_sec: Optional[float]
    domain_strengths: List[str]
    manual_escalation_only: bool = False

@dataclass
class ModelRecommendation:
    rank: int
    model_id: str
    provider: str
    tier: str
    estimated_cost_usd: float
    domain_match_count: int
    reasons: List[str]              # Human-readable: why this model

@dataclass
class RoutingResult:
    prompt_profile: PromptProfile
    resolved_tier: str
    recommendations: List[ModelRecommendation]
    rejections: Dict[str, str]      # model_id -> specific rejection reason
    tier_escalated: bool
    escalation_reason: Optional[str]
    warnings: List[str] = field(default_factory=list)

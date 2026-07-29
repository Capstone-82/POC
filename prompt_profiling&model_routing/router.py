import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

# Fix Windows PyTorch c10.dll loading
try:
    import site
    for site_pkg in site.getsitepackages():
        t_lib = os.path.join(site_pkg, "torch", "lib")
        if os.path.exists(t_lib) and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(t_lib)
except Exception:
    pass

import json
import pickle
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer

from routing_models import (
    PromptProfile,
    ModelCandidate,
    ModelRecommendation,
    RoutingResult,
)
from features import (
    SCORE_COLS,
    VALID_SCORES,
    DIMENSION_LABELS,
    SCORE_TO_CLASS,
    D1_TO_INTENT,
    handcrafted_features,
    extract_research_signals,
    complexity_score_from_dims,
    tier_from_score,
    transform_shared_features,
)

OUTPUT_TOKEN_ESTIMATE = {
    0.00: 200,
    0.25: 500,
    0.50: 1500,
    0.75: 4000,
    1.00: 8000,
}

TIER_RANK = {"T1": 0, "T2": 1, "T3": 2}
RANK_TIER = {0: "T1", 1: "T2", 2: "T3"}


def count_input_tokens(prompt: str) -> int:
    """Count input tokens using tiktoken cl100k_base if installed, else words * 1.3 approximation."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(prompt))
    except Exception:
        words = len(prompt.split())
        return max(1, int(words * 1.3))


class PromptProfiler:
    """Loads the ML profiler pickle bundle and predicts complexity profiles for prompts."""

    def __init__(self, pkl_path: str = "prompt_profiler.pkl"):
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Model bundle not found at {pkl_path}")

        with open(pkl_path, "rb") as f:
            bundle = pickle.load(f)

        self.pca = bundle["pca"]
        self.scaler = bundle["scaler"]
        self.knn_model = bundle["knn_model"]
        self.train_df_subset = bundle["train_df_subset"]
        self.heads = bundle["heads"]
        self.label_encoders = bundle["label_encoders"]
        self.embedding_model_name = bundle.get("embedding_model_name", "BAAI/bge-base-en-v1.5")
        self.embedding_prefix = bundle.get("embedding_query_prefix", "Represent this sentence: ")

        # Deserialise XGBoost models if stored as raw JSON bytes or strings
        self._deserialized_heads = {}
        for name, model_obj in self.heads.items():
            if isinstance(model_obj, (bytes, str)):
                clf = XGBClassifier()
                clf.load_raw(model_obj)
                self._deserialized_heads[name] = clf
            else:
                self._deserialized_heads[name] = model_obj

        # Load embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

    def _build_features(self, prompts: List[str], domains: Optional[List[str]] = None) -> np.ndarray:
        prompts_str = [str(p) for p in prompts]
        prefixed = [self.embedding_prefix + p for p in prompts_str]
        embs = self.embedding_model.encode(
            prefixed, batch_size=32, show_progress_bar=False, normalize_embeddings=True
        )
        hand = handcrafted_features(prompts_str, phrasing_styles=None, domains=domains)
        return transform_shared_features(
            embs, hand.values, self.pca, self.scaler, self.knn_model, self.train_df_subset
        )

    def profile(self, prompt: str, max_tokens: Optional[int] = None) -> PromptProfile:
        input_tokens = count_input_tokens(prompt)

        # 1. Predict domain first without domain dummy leak
        X_temp = self._build_features([prompt], domains=None)
        domain_feat_indices = list(range(130)) + list(range(154, X_temp.shape[1]))
        X_domain = X_temp[:, domain_feat_indices]

        dom_class = self._deserialized_heads["domain"].predict(X_domain)[0]
        domain = self.label_encoders["domain"].inverse_transform([dom_class])[0]

        # 2. Build full feature vector with predicted domain
        X_one = self._build_features([prompt], domains=[domain])

        # 3. Predict classification heads
        reasoning_chain = bool(int(self._deserialized_heads["reasoning_chain_detected"].predict(X_one)[0]))
        intent = self.label_encoders["intent"].inverse_transform(self._deserialized_heads["intent"].predict(X_one))[0]
        task_type = self.label_encoders["task_type"].inverse_transform(self._deserialized_heads["task_type"].predict(X_one))[0]

        # 4. Predict dimension distributions & expected scores
        predicted_dims = {}
        expected_scores = {}
        dim_confidences = []

        for col in SCORE_COLS:
            clf = self._deserialized_heads[col]
            probas = clf.predict_proba(X_one)
            pred_class = np.argmax(probas, axis=1)[0]
            predicted_dims[col] = VALID_SCORES[pred_class]
            dim_confidences.append(float(np.max(probas)))
            expected_scores[col] = float(probas @ np.array(VALID_SCORES))

        score = complexity_score_from_dims(
            expected_scores["d1"],
            expected_scores["d2"],
            expected_scores["d3"],
            expected_scores["d4"],
            expected_scores["d5"],
        )
        derived_tier = tier_from_score(score)

        # Overall confidence = mean of max probas across standard heads & dimension heads
        proba_sums = []
        for name in ["intent", "task_type", "reasoning_chain_detected"]:
            clf = self._deserialized_heads[name]
            proba_sums.append(float(np.max(clf.predict_proba(X_one)[0])))

        key_confidences = dim_confidences + proba_sums
        confidence = float(np.mean(key_confidences))

        # Output token estimation from d3 or caller override
        if max_tokens is not None and max_tokens > 0:
            est_output_tokens = max_tokens
        else:
            est_output_tokens = OUTPUT_TOKEN_ESTIMATE.get(predicted_dims["d3"], 1500)

        research_signals = extract_research_signals(prompt, predicted_dims["d4"])

        return PromptProfile(
            d1=predicted_dims["d1"],
            d2=predicted_dims["d2"],
            d3=predicted_dims["d3"],
            d4=predicted_dims["d4"],
            d5=predicted_dims["d5"],
            domain=domain,
            complexity_score=round(float(score), 4),
            derived_tier=derived_tier,
            intent=intent,
            task_type=task_type,
            reasoning_chain_detected=reasoning_chain,
            research_signals=research_signals,
            confidence=round(confidence, 4),
            input_token_count=input_tokens,
            est_output_tokens=est_output_tokens,
        )


class ModelRegistry:
    """Loads and queries model_registry_v3.json."""

    def __init__(self, json_path: str = "model_registry_v3.json"):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Model registry JSON not found at {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.metadata = data.get("registry_metadata", {})
        self.tier_definitions = data.get("tiers", {
            "T1": {"d1_max": 0.25, "d2_max": 0.5, "d3_max": 0.25},
            "T2": {"d1_max": 0.5, "d2_max": 0.75, "d3_max": 0.75},
            "T3": {"d1_max": 1.0, "d2_max": 1.0, "d3_max": 1.0},
        })
        self.confidence_handling = data.get("confidence_handling", {
            "boundary_buffer": 0.04,
        })

        self.models: List[ModelCandidate] = []
        for m in data.get("models", []):
            candidate = ModelCandidate(
                model_id=m["model_id"],
                provider=m.get("provider", "unknown"),
                generation=m.get("generation", "current"),
                tier=m["tier"],
                cost_in=float(m.get("cost_in", 0.0)),
                cost_out=float(m.get("cost_out", 0.0)),
                max_input_tokens=int(m.get("max_input_tokens", 128000)),
                max_output_tokens=int(m.get("max_output_tokens", 4096)),
                tool_tier=int(m.get("tool_tier", 1)),
                reasoning_mode=bool(m.get("reasoning_mode", False)),
                speed_tokens_per_sec=float(m["speed_tokens_per_sec"]) if m.get("speed_tokens_per_sec") is not None else None,
                domain_strengths=m.get("domain_strengths", []),
                manual_escalation_only=bool(m.get("manual_escalation_only", False)),
            )
            self.models.append(candidate)

    def get_models(self) -> List[ModelCandidate]:
        return self.models


class ModelRouter:
    """Core routing engine implementing the filter-then-sort-by-cost algorithm."""

    def __init__(self, profiler: PromptProfiler, registry: ModelRegistry):
        self.profiler = profiler
        self.registry = registry

    def _resolve_tier(self, profile: PromptProfile) -> Tuple[str, bool, Optional[str]]:
        buffer = self.registry.confidence_handling.get("boundary_buffer", 0.04)
        score = profile.complexity_score
        base_tier = profile.derived_tier

        # Boundary buffer shift
        if abs(score - 0.40) <= buffer and base_tier == "T1":
            base_tier = "T2"
        elif abs(score - 0.70) <= buffer and base_tier == "T2":
            base_tier = "T3"

        # Confidence escalation
        if profile.confidence >= 0.75:
            return base_tier, False, None
        elif profile.confidence >= 0.50:
            new_rank = min(TIER_RANK[base_tier] + 1, 2)
            new_tier = RANK_TIER[new_rank]
            if new_tier != base_tier:
                reason = f"confidence {profile.confidence:.4f} < 0.75 — escalated {base_tier} -> {new_tier}"
                return new_tier, True, reason
            return base_tier, False, None
        else:
            new_rank = min(TIER_RANK[base_tier] + 1, 2)
            new_tier = RANK_TIER[new_rank]
            reason = f"confidence {profile.confidence:.4f} < 0.50 — escalated {base_tier} -> {new_tier}, flagged for review"
            return new_tier, True, reason

    def _filter(self, profile: PromptProfile, resolved_tier: str, include_legacy: bool) -> Tuple[List[ModelCandidate], Dict[str, str]]:
        rejections = {}
        survivors = []
        input_with_margin = int(profile.input_token_count * 1.15)

        for model in self.registry.get_models():
            # Filter 1: Generation gate
            if not include_legacy and model.generation == "legacy":
                rejections[model.model_id] = "generation=legacy excluded by default"
                continue

            # Filter 2: Manual escalation gate
            if model.manual_escalation_only:
                rejections[model.model_id] = "manual_escalation_only=true"
                continue

            # Filter 3: Resolved tier rank check
            if TIER_RANK[model.tier] < TIER_RANK[resolved_tier]:
                rejections[model.model_id] = (
                    f"model tier {model.tier} < resolved tier {resolved_tier}"
                )
                continue

            # Filter 4: Per-dimension ceiling check
            tier_def = self.registry.tier_definitions.get(model.tier, {"d1_max": 1.0, "d2_max": 1.0, "d3_max": 1.0})
            dim_fail = None
            for dim, ceil_key in [("d1", "d1_max"), ("d2", "d2_max"), ("d3", "d3_max")]:
                val = getattr(profile, dim)
                ceiling = tier_def[ceil_key]
                if val > ceiling:
                    dim_fail = f"{dim}={val} exceeds {model.tier} ceiling {ceil_key}={ceiling}"
                    break
            if dim_fail:
                rejections[model.model_id] = dim_fail
                continue

            # Filter 5: Input token capacity check (with 15% safety margin)
            if input_with_margin > model.max_input_tokens:
                rejections[model.model_id] = (
                    f"input {input_with_margin:,} tokens (15% margin) > max_input_tokens {model.max_input_tokens:,}"
                )
                continue

            # Filter 6: Reasoning gate
            if profile.reasoning_chain_detected and not model.reasoning_mode:
                rejections[model.model_id] = "reasoning_chain_detected=true but reasoning_mode=false"
                continue

            survivors.append(model)

        return survivors, rejections

    def _estimate_cost(self, model: ModelCandidate, profile: PromptProfile) -> float:
        input_cost = (profile.input_token_count / 1_000_000.0) * model.cost_in
        output_cost = (profile.est_output_tokens / 1_000_000.0) * model.cost_out
        return round(input_cost + output_cost, 6)

    def _count_domain_match(self, model: ModelCandidate, profile: PromptProfile) -> int:
        match_terms = set()
        match_terms.add(profile.domain.lower().replace(" ", "_"))
        match_terms.add(profile.task_type.lower())
        for sig in profile.research_signals:
            match_terms.add(sig.lower())

        return sum(1 for s in model.domain_strengths if s.lower() in match_terms)

    def _sort_by_cost(self, survivors: List[ModelCandidate], profile: PromptProfile) -> List[ModelCandidate]:
        def sort_key(m: ModelCandidate):
            cost = self._estimate_cost(m, profile)
            domain_match = self._count_domain_match(m, profile)
            speed = m.speed_tokens_per_sec if m.speed_tokens_per_sec is not None else 0.0
            return (cost, -domain_match, -speed)

        return sorted(survivors, key=sort_key)

    def _build_reasons(self, model: ModelCandidate, profile: PromptProfile, resolved_tier: str) -> List[str]:
        reasons = []
        if model.tier == resolved_tier:
            reasons.append(f"Direct tier match ({model.tier})")
        else:
            reasons.append(f"Over-provisioned tier ({model.tier} for {resolved_tier} prompt)")

        if profile.reasoning_chain_detected and model.reasoning_mode:
            reasons.append("Supports required reasoning mode")

        match_count = self._count_domain_match(model, profile)
        if match_count > 0:
            reasons.append(f"Matched {match_count} domain strength tag(s)")

        return reasons

    def _collect_warnings(self, profile: PromptProfile, survivors: List[ModelCandidate]) -> List[str]:
        warnings = []
        if not survivors:
            warnings.append("No models passed all filters for this prompt.")
        if profile.confidence < 0.50:
            warnings.append(f"Low confidence profile ({profile.confidence:.4f}). Flagged for manual review.")
        return warnings

    def route(self, prompt: str, max_tokens: Optional[int] = None, include_legacy: bool = False, top_n: int = 3) -> RoutingResult:
        profile = self.profiler.profile(prompt, max_tokens=max_tokens)
        resolved_tier, escalated, escalation_reason = self._resolve_tier(profile)
        survivors, rejections = self._filter(profile, resolved_tier, include_legacy)
        ranked = self._sort_by_cost(survivors, profile)

        recommendations = []
        for i, model in enumerate(ranked[:top_n]):
            recommendations.append(
                ModelRecommendation(
                    rank=i + 1,
                    model_id=model.model_id,
                    provider=model.provider,
                    tier=model.tier,
                    estimated_cost_usd=self._estimate_cost(model, profile),
                    domain_match_count=self._count_domain_match(model, profile),
                    reasons=self._build_reasons(model, profile, resolved_tier),
                )
            )

        return RoutingResult(
            prompt_profile=profile,
            resolved_tier=resolved_tier,
            recommendations=recommendations,
            rejections=rejections,
            tier_escalated=escalated,
            escalation_reason=escalation_reason,
            warnings=self._collect_warnings(profile, survivors),
        )


def route_model(
    prompt: str,
    max_tokens: Optional[int] = None,
    pkl_path: str = "prompt_profiler.pkl",
    registry_path: str = "model_registry_v3.json",
    include_legacy: bool = False,
    top_n: int = 3,
) -> RoutingResult:
    """One-shot convenience routing function."""
    profiler = PromptProfiler(pkl_path)
    registry = ModelRegistry(registry_path)
    router = ModelRouter(profiler, registry)
    return router.route(prompt, max_tokens=max_tokens, include_legacy=include_legacy, top_n=top_n)

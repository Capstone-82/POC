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
from typing import Any, List, Dict, Tuple, Optional
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

DEFAULT_WEIGHTED_ROUTING = {
    "weights": {
        "T1": {"quality": 25, "capability": 25, "context": 10, "reliability": 10, "latency": 15, "cost": 15},
        "T2": {"quality": 35, "capability": 25, "context": 10, "reliability": 15, "latency": 10, "cost": 5},
        "T3": {"quality": 45, "capability": 25, "context": 10, "reliability": 15, "latency": 0, "cost": 5},
        "T3_critical": {"quality": 50, "capability": 25, "context": 10, "reliability": 15, "latency": 0, "cost": 0},
    },
    "quality_floors": {"T1": 60, "T2": 70, "T3": 80, "T3_critical": 88},
    "context_safety_margin": 0.15,
    "near_tie_points": 3.0,
    "low_confidence_threshold": 0.50,
}


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
        self.quality_first_routing = data.get("quality_first_routing", {
            "enabled": True,
            "quality_target_model": "claude-opus-5",
            "min_d1_threshold": 0.75,
            "strategic_intent_only": True,
            "high_dependency_dimensions": ["d4", "d5"],
            "min_dependency_score": 1.0,
            "require_reasoning_chain": True
        })
        self.weighted_routing = data.get("weighted_routing", DEFAULT_WEIGHTED_ROUTING)
        self.provider_metadata = data.get("providers", {})

        self.models: List[ModelCandidate] = []
        for m in data.get("models", []):
            candidate = ModelCandidate(
                model_id=m["model_id"],
                provider=m.get("provider", "unknown"),
                generation=m.get("generation", "current"),
                tier=m["tier"],
                cost_in=float(m.get("cost_in") if m.get("cost_in") is not None else 0.0),
                cost_out=float(m.get("cost_out") if m.get("cost_out") is not None else 0.0),
                max_input_tokens=int(m.get("max_input_tokens", 128000)),
                max_output_tokens=int(m.get("max_output_tokens", 4096)),
                tool_tier=int(m.get("tool_tier", 1)),
                reasoning_mode=bool(m.get("reasoning_mode", False)),
                speed_tokens_per_sec=float(m["speed_tokens_per_sec"]) if m.get("speed_tokens_per_sec") is not None else None,
                domain_strengths=m.get("domain_strengths", []),
                manual_escalation_only=bool(m.get("manual_escalation_only", False)),
                total_context_tokens=int(m.get("total_context_tokens", m.get("max_input_tokens", 128000))),
                api_model_id=m.get("api_model_id"),
                lifecycle_status=m.get("lifecycle", {}).get(
                    "status", "deprecated" if m.get("generation") == "legacy" else "active"
                ),
                verification_status=m.get("verification_status", "Needs Manual Verification"),
                capability_tags=m.get("capability_tags", m.get("domain_strengths", [])),
                latency_slo_ms=m.get("latency_slo_ms"),
                availability_status=m.get("availability_status", "unknown"),
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

    def _filter(
        self,
        profile: PromptProfile,
        resolved_tier: str,
        include_legacy: bool,
        required_capabilities: List[str],
    ) -> Tuple[List[ModelCandidate], Dict[str, str]]:
        rejections = {}
        survivors = []
        margin = float(self.registry.weighted_routing.get("context_safety_margin", 0.15))
        input_with_margin = int(profile.input_token_count * (1 + margin))
        total_request_tokens = input_with_margin + profile.est_output_tokens

        for model in self.registry.get_models():
            # Lifecycle is authoritative when present; generation remains a compatibility fallback.
            if model.lifecycle_status in {"deprecated", "retired", "disabled"}:
                rejections[model.model_id] = f"lifecycle.status={model.lifecycle_status}"
                continue
            if not include_legacy and model.generation == "legacy":
                rejections[model.model_id] = "generation=legacy excluded by default"
                continue

            # Filter 2: Manual escalation gate
            if model.manual_escalation_only:
                rejections[model.model_id] = "manual_escalation_only=true"
                continue

            if model.availability_status in {"unavailable", "degraded"}:
                rejections[model.model_id] = f"availability_status={model.availability_status}"
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

            if profile.est_output_tokens > model.max_output_tokens:
                rejections[model.model_id] = (
                    f"requested output {profile.est_output_tokens:,} > max_output_tokens {model.max_output_tokens:,}"
                )
                continue

            if total_request_tokens > (model.total_context_tokens or model.max_input_tokens):
                rejections[model.model_id] = (
                    f"total request {total_request_tokens:,} > total_context_tokens "
                    f"{(model.total_context_tokens or model.max_input_tokens):,}"
                )
                continue

            missing_capabilities = [
                capability for capability in required_capabilities
                if capability.lower() not in {tag.lower() for tag in model.capability_tags}
            ]
            if missing_capabilities:
                rejections[model.model_id] = f"missing required capabilities: {', '.join(missing_capabilities)}"
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

    def _score_candidate(
        self,
        model: ModelCandidate,
        profile: PromptProfile,
        resolved_tier: str,
        enterprise_criticality: str,
        costs: List[float],
    ) -> Dict[str, float]:
        """Produce an explainable 0-100 weighted score after hard feasibility gates."""
        critical = enterprise_criticality.lower() in {"high", "critical", "regulated", "safety_critical"}
        policy_key = "T3_critical" if resolved_tier == "T3" and critical else resolved_tier
        weights = self.registry.weighted_routing.get("weights", DEFAULT_WEIGHTED_ROUTING["weights"]).get(
            policy_key, DEFAULT_WEIGHTED_ROUTING["weights"][policy_key]
        )

        tier_fit = 80.0 if model.tier == resolved_tier else 92.0
        reasoning_fit = 15.0 if profile.reasoning_chain_detected and model.reasoning_mode else 0.0
        complexity_fit = 5.0 * sum(1 for value in [profile.d1, profile.d2, profile.d3, profile.d4, profile.d5] if value >= 0.75)
        quality = min(100.0, tier_fit + reasoning_fit + complexity_fit)

        domain_match = self._count_domain_match(model, profile)
        capability_terms = {tag.lower() for tag in model.capability_tags}
        signal_bonus = 0.0
        if profile.d4 >= 0.75 and any("research" in tag or "ground" in tag for tag in capability_terms):
            signal_bonus += 10.0
        if profile.d5 >= 0.75 and any("context" in tag for tag in capability_terms):
            signal_bonus += 10.0
        if profile.intent.lower() in capability_terms:
            signal_bonus += 10.0
        capability = min(100.0, 55.0 + domain_match * 15.0 + signal_bonus + (10.0 if profile.reasoning_chain_detected and model.reasoning_mode else 0.0))

        total_context = model.total_context_tokens or model.max_input_tokens
        request_tokens = int(profile.input_token_count * (1 + float(self.registry.weighted_routing.get("context_safety_margin", 0.15)))) + profile.est_output_tokens
        context = max(0.0, min(100.0, 100.0 * (total_context - request_tokens) / max(1, total_context)))
        reliability = 100.0 if model.availability_status == "available" else 70.0
        latency = 60.0 if model.speed_tokens_per_sec is None else min(100.0, model.speed_tokens_per_sec / 2.0)

        cost = self._estimate_cost(model, profile)
        low, high = min(costs), max(costs)
        cost_efficiency = 50.0 if high == low else 100.0 * (high - cost) / (high - low)

        components = {
            "quality": quality,
            "capability": capability,
            "context": context,
            "reliability": reliability,
            "latency": latency,
            "cost": cost_efficiency,
        }
        weighted = sum(components[name] * float(weights.get(name, 0)) / 100.0 for name in components)
        components["routing_score"] = round(weighted, 2)
        components["quality_floor"] = float(self.registry.weighted_routing.get("quality_floors", DEFAULT_WEIGHTED_ROUTING["quality_floors"]).get(policy_key, 0))
        return components

    def _rank_weighted(
        self,
        survivors: List[ModelCandidate],
        profile: PromptProfile,
        resolved_tier: str,
        enterprise_criticality: str,
    ) -> List[Tuple[ModelCandidate, Dict[str, float]]]:
        costs = [self._estimate_cost(model, profile) for model in survivors]
        scored = [
            (model, self._score_candidate(model, profile, resolved_tier, enterprise_criticality, costs))
            for model in survivors
        ]
        eligible = [(model, score) for model, score in scored if score["quality"] >= score["quality_floor"]]
        candidates = eligible or scored
        return sorted(
            candidates,
            key=lambda item: (
                -item[1]["routing_score"], -item[1]["quality"], -item[1]["reliability"],
                -item[1]["context"], -item[1]["latency"], self._estimate_cost(item[0], profile), item[0].model_id,
            ),
        )

    def _build_reasons(
        self, model: ModelCandidate, profile: PromptProfile, resolved_tier: str, score: Optional[Dict[str, float]] = None
    ) -> List[str]:
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

        if score is not None:
            reasons.append(
                f"Weighted routing score {score['routing_score']:.2f} "
                f"(quality {score['quality']:.0f}, capability {score['capability']:.0f}, "
                f"context {score['context']:.0f}, reliability {score['reliability']:.0f})"
            )

        return reasons

    def _collect_warnings(self, profile: PromptProfile, survivors: List[ModelCandidate]) -> List[str]:
        warnings = []
        if not survivors:
            warnings.append("No models passed all filters for this prompt.")
        if profile.confidence < float(self.registry.weighted_routing.get("low_confidence_threshold", 0.50)):
            warnings.append(f"Low confidence profile ({profile.confidence:.4f}). Flagged for manual review.")
        return warnings

    def route(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        include_legacy: bool = False,
        top_n: int = 3,
        enterprise_criticality: str = "standard",
        required_capabilities: Optional[List[str]] = None,
    ) -> RoutingResult:
        profile = self.profiler.profile(prompt, max_tokens=max_tokens)
        resolved_tier, escalated, escalation_reason = self._resolve_tier(profile)
        survivors, rejections = self._filter(
            profile, resolved_tier, include_legacy, required_capabilities or []
        )
        ranked = self._rank_weighted(survivors, profile, resolved_tier, enterprise_criticality) if survivors else []

        recommendations = []
        for i, (model, score) in enumerate(ranked[:top_n]):
            reasons = self._build_reasons(model, profile, resolved_tier, score)
            recommendations.append(
                ModelRecommendation(
                    rank=i + 1,
                    model_id=model.model_id,
                    provider=model.provider,
                    tier=model.tier,
                    estimated_cost_usd=self._estimate_cost(model, profile),
                    domain_match_count=self._count_domain_match(model, profile),
                    reasons=reasons,
                    routing_score=score["routing_score"],
                    score_breakdown=score,
                )
            )

        warnings = self._collect_warnings(profile, survivors)
        if profile.confidence < float(self.registry.weighted_routing.get("low_confidence_threshold", 0.50)):
            warnings.append("Conservative tier escalation applied because profile confidence is low.")
        if len(ranked) > 1:
            margin = float(self.registry.weighted_routing.get("near_tie_points", 3.0))
            if ranked[0][1]["routing_score"] - ranked[1][1]["routing_score"] <= margin:
                warnings.append("Top candidates are near-tied; deterministic quality, reliability, context, latency, and cost tie-breakers applied.")

        return RoutingResult(
            prompt_profile=profile,
            resolved_tier=resolved_tier,
            recommendations=recommendations,
            rejections=rejections,
            tier_escalated=escalated,
            escalation_reason=escalation_reason,
            warnings=warnings,
        )


def route_model(
    prompt: str,
    max_tokens: Optional[int] = None,
    pkl_path: str = "prompt_profiler.pkl",
    registry_path: str = "model_registry_v3.json",
    include_legacy: bool = False,
    top_n: int = 3,
    enterprise_criticality: str = "standard",
    required_capabilities: Optional[List[str]] = None,
) -> RoutingResult:
    """One-shot convenience routing function."""
    profiler = PromptProfiler(pkl_path)
    registry = ModelRegistry(registry_path)
    router = ModelRouter(profiler, registry)
    return router.route(
        prompt, max_tokens=max_tokens, include_legacy=include_legacy, top_n=top_n,
        enterprise_criticality=enterprise_criticality, required_capabilities=required_capabilities,
    )

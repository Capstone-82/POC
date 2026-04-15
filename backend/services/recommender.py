from __future__ import annotations

import math
import pickle
import re
import csv
import asyncio
import uuid
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from services.supabase_client import get_benchmark_data, get_prompt_logs

VALID_COMPLEXITIES = {"low", "mid", "high"}
VALID_CLARITIES = {"CLEAR", "PARTIAL", "UNCLEAR"}

MIN_SAMPLES_PER_MODEL = 5
ACCURACY_TOLERANCE = 2.0
MIN_ACCURACY_GAIN = 2.0
MIN_COST_IMPROVEMENT_PCT = 15.0
MIN_LATENCY_IMPROVEMENT_PCT = 20.0
MIN_KNN_NEIGHBORS = 5

CLASSIFIER_PATH = (
    Path(__file__).resolve().parents[2] / "model_training" / "artifacts" / "classifier.pkl"
)
LOCAL_BENCHMARK_CSV = Path(__file__).resolve().parents[2] / "model_training" / "benchmark_results.csv"
LOCAL_PROMPT_LOGS_CSV = Path(__file__).resolve().parents[2] / "model_training" / "prompt_logs_rows.csv"


def normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip().lower())


def build_classifier_input(prompt: str, use_case: str) -> str:
    return f"use_case: {use_case.strip().lower()}\nprompt: {prompt.strip()}"


def load_local_csv_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


async def load_benchmark_rows_with_fallback(use_case: Optional[str] = None) -> Tuple[List[dict], str]:
    try:
        supabase_rows = await get_benchmark_data(use_case=use_case)
    except Exception:
        supabase_rows = []

    cleaned_supabase_rows = clean_benchmark_rows(supabase_rows)
    if cleaned_supabase_rows:
        return cleaned_supabase_rows, "supabase"

    local_rows = clean_benchmark_rows(load_local_csv_rows(LOCAL_BENCHMARK_CSV))
    if use_case:
        local_rows = [row for row in local_rows if row["use_case"] == use_case]
    return local_rows, "local_csv"


def load_complexity_classifier():
    if not CLASSIFIER_PATH.exists():
        return None
    with CLASSIFIER_PATH.open("rb") as file:
        return pickle.load(file)


def infer_complexity(prompt: str, use_case: str, classifier: Optional[Any]) -> Tuple[str, Optional[float], str]:
    classifier_input = build_classifier_input(prompt, use_case)
    if classifier is not None:
        prediction = str(classifier.predict([classifier_input])[0]).strip().lower()
        confidence = None
        if hasattr(classifier, "predict_proba"):
            try:
                confidence = float(max(classifier.predict_proba([classifier_input])[0]))
            except Exception:
                confidence = None
        if prediction in VALID_COMPLEXITIES:
            return prediction, confidence, "classifier"

    prompt_lc = prompt.lower()
    word_count = len(re.findall(r"\w+", prompt_lc))
    if word_count <= 10 and not any(
        token in prompt_lc
        for token in (
            "explain",
            "compare",
            "design",
            "architecture",
            "optimize",
            "analyze",
            "tradeoff",
            "distributed",
            "debug",
            "production",
        )
    ):
        return "low", None, "heuristic"
    if any(
        token in prompt_lc
        for token in (
            "distributed",
            "byzantine",
            "architecture",
            "multi-tenant",
            "production-ready",
            "fault tolerance",
            "tradeoff",
            "benchmark",
            "optimize",
            "design a system",
        )
    ) or word_count >= 45:
        return "high", None, "heuristic"
    return "mid", None, "heuristic"


async def infer_clarity(prompt: str, use_case: str) -> Tuple[str, str]:
    try:
        exact_logs = await get_prompt_logs(use_case=use_case, prompt=prompt)
    except Exception:
        exact_logs = []
    if exact_logs:
        counts: dict[str, int] = {}
        for row in exact_logs:
            label = str(row.get("clarity", "")).strip().upper()
            if label in VALID_CLARITIES:
                counts[label] = counts.get(label, 0) + 1
        if counts:
            top_label = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
            return top_label, "prompt_logs_exact"

    normalized = normalize_prompt(prompt)
    local_prompt_logs = load_local_csv_rows(LOCAL_PROMPT_LOGS_CSV)
    local_counts: Dict[str, int] = {}
    for row in local_prompt_logs:
        row_use_case = str(row.get("use_case", "")).strip().lower()
        row_prompt = str(row.get("prompt", ""))
        row_clarity = str(row.get("clarity", "")).strip().upper()
        if row_use_case == use_case and normalize_prompt(row_prompt) == normalized and row_clarity in VALID_CLARITIES:
            local_counts[row_clarity] = local_counts.get(row_clarity, 0) + 1
    if local_counts:
        top_label = sorted(local_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        return top_label, "prompt_logs_local_exact"

    word_count = len(re.findall(r"\w+", prompt))
    prompt_lc = prompt.lower().strip()

    if word_count <= 3:
        return "UNCLEAR", "heuristic"

    ambiguous_markers = [
        "make it better",
        "do this",
        "fix this",
        "improve this",
        "something",
        "etc",
        "whatever",
    ]
    if any(marker in prompt_lc for marker in ambiguous_markers):
        return "UNCLEAR", "heuristic"

    explicit_verbs = [
        "write",
        "create",
        "generate",
        "explain",
        "summarize",
        "compare",
        "implement",
        "build",
        "design",
        "solve",
        "calculate",
        "analyze",
    ]
    constraint_markers = [
        "with",
        "using",
        "for",
        "include",
        "return",
        "without",
        "in python",
        "in java",
        "step by step",
    ]
    has_explicit_task = any(token in prompt_lc for token in explicit_verbs)
    has_constraints = any(token in prompt_lc for token in constraint_markers)

    if has_explicit_task and (has_constraints or word_count >= 8):
        return "CLEAR", "heuristic"
    if has_explicit_task or word_count >= 6:
        return "PARTIAL", "heuristic"
    return "UNCLEAR", "heuristic"


def clean_benchmark_rows(rows: List[dict]) -> List[dict]:
    cleaned: List[dict] = []
    for row in rows:
        try:
            model_id   = str(row["model_id"]).strip()
            provider   = str(row.get("provider", "")).strip()
            use_case   = str(row["use_case"]).strip().lower()
            complexity = str(row["prompt_complexity"]).strip().lower()
            clarity    = str(row["clarity"]).strip().upper()

            avg_acc = row.get("avg_accuracy_score")
            leg_acc = row.get("accuracy_score")

            if avg_acc is not None and str(avg_acc).strip() != "":
                try:
                    accuracy_score = float(avg_acc)
                except (TypeError, ValueError):
                    accuracy_score = None
            elif leg_acc is not None and str(leg_acc).strip() != "":
                try:
                    accuracy_score = float(leg_acc)
                except (TypeError, ValueError):
                    accuracy_score = None
            else:
                continue

            if accuracy_score is None:
                continue

            cost       = float(row["cost"])
            latency_ms = float(row["latency_ms"])
        except (KeyError, TypeError, ValueError):
            continue

        if not model_id or use_case == "" or complexity not in VALID_COMPLEXITIES or clarity not in VALID_CLARITIES:
            continue

        cleaned.append(
            {
                "model_id": model_id,
                "provider": provider,
                "use_case": use_case,
                "prompt_complexity": complexity,
                "clarity": clarity,
                "accuracy_score": accuracy_score,
                "cost": cost,
                "latency_ms": latency_ms,
                "has_multi_eval": avg_acc is not None and str(avg_acc).strip() != "",
            }
        )
    return cleaned


def summarize_models(rows: List[dict]) -> List[dict]:
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (row["model_id"], row["provider"])
        bucket = grouped.setdefault(
            key,
            {
                "model_id": row["model_id"],
                "provider": row["provider"],
                "accuracy_scores": [],
                "costs": [],
                "latencies": [],
            },
        )
        bucket["accuracy_scores"].append(float(row["accuracy_score"]))
        bucket["costs"].append(float(row["cost"]))
        bucket["latencies"].append(float(row["latency_ms"]))

    summaries: List[dict] = []
    for stats in grouped.values():
        sample_count = len(stats["accuracy_scores"])
        if sample_count < MIN_SAMPLES_PER_MODEL:
            continue
        summaries.append(
            {
                "model_id": stats["model_id"],
                "provider": stats["provider"],
                "sample_count": sample_count,
                "avg_accuracy": sum(stats["accuracy_scores"]) / sample_count,
                "median_accuracy": float(median(stats["accuracy_scores"])),
                "median_cost": float(median(stats["costs"])),
                "median_latency_ms": float(median(stats["latencies"])),
            }
        )

    summaries.sort(
        key=lambda item: (
            -item["avg_accuracy"],
            -item["sample_count"],
            item["median_cost"],
            item["median_latency_ms"],
        )
    )
    return summaries


def normalize_lower_better(value: float, min_value: float, max_value: float) -> float:
    if math.isclose(min_value, max_value):
        return 1.0
    return 1.0 - ((value - min_value) / (max_value - min_value))


def pick_best_value_model(summary: List[dict]) -> dict:
    top_accuracy = max(item["avg_accuracy"] for item in summary)
    shortlist = [item for item in summary if item["avg_accuracy"] >= top_accuracy - ACCURACY_TOLERANCE]

    cost_min = min(item["median_cost"] for item in shortlist)
    cost_max = max(item["median_cost"] for item in shortlist)
    latency_min = min(item["median_latency_ms"] for item in shortlist)
    latency_max = max(item["median_latency_ms"] for item in shortlist)

    for item in shortlist:
        item["value_score"] = (
            0.75 * normalize_lower_better(item["median_cost"], cost_min, cost_max)
            + 0.25
            * normalize_lower_better(item["median_latency_ms"], latency_min, latency_max)
        )

    shortlist.sort(
        key=lambda item: (-item["value_score"], -item["avg_accuracy"], -item["sample_count"])
    )
    return shortlist[0]


def percent_delta(new_value: float, old_value: float) -> Optional[float]:
    if old_value == 0:
        return None
    return ((new_value - old_value) / old_value) * 100.0


def build_model_stats(row: dict) -> dict:
    return {
        "model_id": row["model_id"],
        "provider": row["provider"],
        "sample_count": int(row["sample_count"]),
        "avg_accuracy": round(float(row["avg_accuracy"]), 2),
        "median_accuracy": round(float(row["median_accuracy"]), 2),
        "median_cost": round(float(row["median_cost"]), 6),
        "median_latency_ms": round(float(row["median_latency_ms"]), 1),
    }


def build_knn_model_stats(row: dict) -> dict:
    accuracy = float(row.get("sim_weighted_accuracy", row.get("avg_accuracy", 0.0)))
    return {
        "model_id": row["model_id"],
        "provider": row.get("provider", ""),
        "sample_count": int(row.get("sample_n", row.get("sample_count", 0))),
        "avg_accuracy": round(accuracy, 2),
        "median_accuracy": round(accuracy, 2),
        "median_cost": round(float(row.get("p50_cost", row.get("median_cost", 0.0))), 6),
        "median_latency_ms": round(float(row.get("p50_latency", row.get("median_latency_ms", 0.0))), 1),
    }


def score_and_rank_knn_candidates(candidates: Dict[str, dict]) -> List[dict]:
    ranked = [dict(item) for item in candidates.values()]
    if not ranked:
        return []

    acc_values = [float(item["sim_weighted_accuracy"]) for item in ranked]
    cost_values = [float(item["p50_cost"]) for item in ranked]
    latency_values = [float(item["p50_latency"]) for item in ranked]
    variance_values = [float(item.get("score_variance", 0.0)) for item in ranked]

    acc_min, acc_max = min(acc_values), max(acc_values)
    cost_min, cost_max = min(cost_values), max(cost_values)
    latency_min, latency_max = min(latency_values), max(latency_values)
    variance_min, variance_max = min(variance_values), max(variance_values)

    for item in ranked:
        acc_norm = 1.0 if math.isclose(acc_min, acc_max) else (
            (float(item["sim_weighted_accuracy"]) - acc_min) / (acc_max - acc_min)
        )
        cost_norm = normalize_lower_better(float(item["p50_cost"]), cost_min, cost_max)
        latency_norm = normalize_lower_better(float(item["p50_latency"]), latency_min, latency_max)
        confidence = normalize_lower_better(
            float(item.get("score_variance", 0.0)),
            variance_min,
            variance_max,
        )
        item["confidence"] = round(confidence, 3)
        item["value_score"] = round(
            0.55 * acc_norm
            + 0.25 * cost_norm
            + 0.15 * latency_norm
            + 0.05 * confidence,
            4,
        )

    ranked.sort(
        key=lambda item: (
            -item["value_score"],
            -float(item["sim_weighted_accuracy"]),
            float(item["p50_cost"]),
            float(item["p50_latency"]),
        )
    )
    return ranked


def should_switch(
    recommended: dict,
    current: Optional[dict],
    min_accuracy_gain: float = 0.0,
    max_cost_increase_pct: float = 0.0,
    max_latency_increase_pct: float = 0.0,
) -> Tuple[bool, str]:
    if current is None:
        return True, "No current model comparison was available, so this is the best value option in the matched benchmark slice."

    accuracy_gain = recommended["accuracy_delta"]
    cost_delta_pct = recommended["cost_delta_pct"]
    latency_delta_pct = recommended["latency_delta_pct"]

    if (
        accuracy_gain is not None
        and accuracy_gain >= min_accuracy_gain
        and cost_delta_pct is not None
        and cost_delta_pct <= max_cost_increase_pct
        and latency_delta_pct is not None
        and latency_delta_pct <= max_latency_increase_pct
    ):
        return (
            True,
            "Candidate meets the selected thresholds for accuracy gain, cost, and latency.",
        )

    return (
        False,
        "Switching requires a candidate to meet the selected accuracy, cost, and latency thresholds.",
    )


def build_reason(
    switch_recommended: bool,
    policy_reason: str,
    recommended_stats: dict,
    current_model: str,
    current_model_found: bool,
    filter_level: str,
) -> str:
    recommended_name = f"{recommended_stats['provider']}/{recommended_stats['model_id']}"
    benchmark_scope = "semantic KNN neighbors" if filter_level == "semantic_knn" else f"{filter_level} benchmark slice"
    if not current_model_found:
        return (
            f"{recommended_name} is the best value option in the {benchmark_scope}. "
            f"{policy_reason}"
        )

    if switch_recommended:
        return (
            f"Switch from {current_model} to {recommended_stats['model_id']}. "
            f"{policy_reason}"
        )

    return (
        f"Stay on {current_model}. {policy_reason} "
        f"The best alternative in the {benchmark_scope} was {recommended_name}."
    )


async def build_slice_recommendation(
    use_case: str,
    prompt: str,
    current_model: str,
    complexity: str,
    complexity_confidence: Optional[float],
    complexity_source: str,
    clarity: str,
    clarity_source: str,
    min_accuracy_gain: float = 0.0,
    max_cost_increase_pct: float = 0.0,
    max_latency_increase_pct: float = 0.0,
    fallback_warning: Optional[str] = None,
) -> dict:
    all_rows, data_source = await load_benchmark_rows_with_fallback(use_case=use_case)
    if not all_rows:
        raise ValueError("No benchmark data was found for this use case.")

    filter_tiers = [
        (
            "exact",
            lambda row: row["prompt_complexity"] == complexity and row["clarity"] == clarity,
        ),
        (
            "use_case_plus_complexity",
            lambda row: row["prompt_complexity"] == complexity,
        ),
        (
            "use_case_only",
            lambda row: True,
        ),
    ]

    candidate_rows: List[dict] = []
    filter_level = "none"
    summary: List[dict] = []
    for tier_name, predicate in filter_tiers:
        tier_rows = [row for row in all_rows if predicate(row)]
        tier_summary = summarize_models(tier_rows)
        if tier_summary:
            candidate_rows = tier_rows
            summary = tier_summary
            filter_level = tier_name
            break

    if not summary:
        raise ValueError("No sufficiently supported benchmark slice was found for this prompt.")

    best_row = pick_best_value_model(summary)
    recommended_stats = build_model_stats(best_row)

    current_row = next((row for row in summary if row["model_id"] == current_model), None)
    current_stats = build_model_stats(current_row) if current_row else None
    current_model_found = current_stats is not None

    recommended_stats["accuracy_delta"] = (
        None if current_stats is None else round(recommended_stats["avg_accuracy"] - current_stats["avg_accuracy"], 2)
    )
    recommended_stats["accuracy_delta_pct"] = (
        None
        if current_stats is None
        else round(
            percent_delta(recommended_stats["avg_accuracy"], current_stats["avg_accuracy"]),
            1,
        )
    )
    recommended_stats["cost_delta_pct"] = (
        None
        if current_stats is None
        else round(
            percent_delta(recommended_stats["median_cost"], current_stats["median_cost"]),
            1,
        )
    )
    recommended_stats["latency_delta_pct"] = (
        None
        if current_stats is None
        else round(
            percent_delta(
                recommended_stats["median_latency_ms"],
                current_stats["median_latency_ms"],
            ),
            1,
        )
    )

    switch_recommended, policy_reason = should_switch(
        recommended_stats,
        current_stats,
        min_accuracy_gain=min_accuracy_gain,
        max_cost_increase_pct=max_cost_increase_pct,
        max_latency_increase_pct=max_latency_increase_pct,
    )
    final_suggestion_model = recommended_stats["model_id"] if switch_recommended else current_model
    reason = build_reason(
        switch_recommended=switch_recommended,
        policy_reason=policy_reason,
        recommended_stats=recommended_stats,
        current_model=current_model,
        current_model_found=current_model_found,
        filter_level=filter_level,
    )

    warnings: List[str] = []
    if fallback_warning:
        warnings.append(fallback_warning)
    if not current_model_found:
        warnings.append(
            f"Current model '{current_model}' was not found in the matched benchmark slice, so comparison deltas were skipped."
        )

    return {
        "complexity": complexity,
        "complexity_confidence": round(complexity_confidence, 3) if complexity_confidence is not None else None,
        "complexity_source": complexity_source,
        "quality_score": None,
        "use_case": use_case,
        "clarity": clarity,
        "clarity_source": clarity_source,
        "filter_level": filter_level,
        "recommendation_mode": "best_value",
        "data_source": data_source,
        "current_model": current_model,
        "current_model_found": current_model_found,
        "current_model_stats": current_stats,
        "recommended_model": recommended_stats["model_id"],
        "recommended_provider": recommended_stats["provider"],
        "expected_accuracy": recommended_stats["avg_accuracy"],
        "expected_cost": recommended_stats["median_cost"],
        "expected_latency": recommended_stats["median_latency_ms"],
        "accuracy_delta": recommended_stats["accuracy_delta"],
        "accuracy_delta_pct": recommended_stats["accuracy_delta_pct"],
        "cost_delta_pct": recommended_stats["cost_delta_pct"],
        "latency_delta_pct": recommended_stats["latency_delta_pct"],
        "sample_size": recommended_stats["sample_count"],
        "slice_row_count": len(candidate_rows),
        "models_considered": len(summary),
        "switch_recommended": switch_recommended,
        "final_suggestion_model": final_suggestion_model,
        "policy_reason": policy_reason,
        "reason": reason,
        "top_candidates": [build_model_stats(row) for row in summary[:5]],
        "warnings": warnings,
        "policy_thresholds": {
            "min_accuracy_gain": min_accuracy_gain,
            "max_cost_increase_pct": max_cost_increase_pct,
            "max_latency_increase_pct": max_latency_increase_pct,
        },
    }


async def get_recommendation(
    use_case: str,
    prompt: str,
    current_model: str,
    min_accuracy_gain: float = 0.0,
    max_cost_increase_pct: float = 0.0,
    max_latency_increase_pct: float = 0.0,
) -> dict:
    classifier = load_complexity_classifier()
    complexity, complexity_confidence, complexity_source = infer_complexity(prompt, use_case, classifier)
    clarity, clarity_source = await infer_clarity(prompt, use_case)

    try:
        return await build_knn_recommendation(
            use_case=use_case,
            prompt=prompt,
            current_model=current_model,
            complexity=complexity,
            complexity_confidence=complexity_confidence,
            complexity_source=complexity_source,
            clarity=clarity,
            clarity_source=clarity_source,
            min_accuracy_gain=min_accuracy_gain,
            max_cost_increase_pct=max_cost_increase_pct,
            max_latency_increase_pct=max_latency_increase_pct,
        )
    except Exception as exc:
        return await build_slice_recommendation(
            use_case=use_case,
            prompt=prompt,
            current_model=current_model,
            complexity=complexity,
            complexity_confidence=complexity_confidence,
            complexity_source=complexity_source,
            clarity=clarity,
            clarity_source=clarity_source,
            min_accuracy_gain=min_accuracy_gain,
            max_cost_increase_pct=max_cost_increase_pct,
            max_latency_increase_pct=max_latency_increase_pct,
            fallback_warning=f"KNN recommendation failed, so slice fallback was used: {exc}",
        )


async def build_knn_recommendation(
    use_case: str,
    prompt: str,
    current_model: str,
    complexity: str,
    complexity_confidence: Optional[float],
    complexity_source: str,
    clarity: str,
    clarity_source: str,
    min_accuracy_gain: float = 0.0,
    max_cost_increase_pct: float = 0.0,
    max_latency_increase_pct: float = 0.0,
) -> dict:
    from services.embedding_service import get_or_compute_embedding
    from services.knn_search import (
        AGGREGATION_FALLBACK_K,
        FALLBACK_K,
        FALLBACK_SIMILARITY,
        aggregate_knn_signals,
        search_neighbors,
    )
    from services.model_registry import get_model_ids_for_use_case
    from services.supabase_client import supabase

    vector, prompt_hash, was_cached = await get_or_compute_embedding(prompt, supabase)
    neighbors = search_neighbors(supabase, vector, use_case)
    if len(neighbors) < MIN_KNN_NEIGHBORS:
        neighbors = search_neighbors(
            supabase,
            vector,
            use_case,
            k=FALLBACK_K,
            min_similarity=FALLBACK_SIMILARITY,
        )

    if len(neighbors) < MIN_KNN_NEIGHBORS:
        raise ValueError(f"KNN found only {len(neighbors)} usable neighbors")

    allowed_models = get_model_ids_for_use_case(use_case)
    def aggregate_allowed(rows: List[dict]) -> Dict[str, dict]:
        return {
            model_id: signals
            for model_id, signals in aggregate_knn_signals(rows).items()
            if model_id in allowed_models
        }

    knn_signals = aggregate_allowed(neighbors)
    if not knn_signals:
        neighbors = search_neighbors(
            supabase,
            vector,
            use_case,
            k=AGGREGATION_FALLBACK_K,
            min_similarity=FALLBACK_SIMILARITY,
        )
        knn_signals = aggregate_allowed(neighbors)

    ranked = score_and_rank_knn_candidates(knn_signals)
    if not ranked:
        raise ValueError("KNN neighbors were too sparse after per-model aggregation")

    best = ranked[0]
    recommended_stats = build_knn_model_stats(best)
    current_row = next((row for row in ranked if row["model_id"] == current_model), None)
    current_stats = build_knn_model_stats(current_row) if current_row else None
    current_model_found = current_stats is not None

    recommended_stats["accuracy_delta"] = (
        None if current_stats is None else round(recommended_stats["avg_accuracy"] - current_stats["avg_accuracy"], 2)
    )
    recommended_stats["accuracy_delta_pct"] = (
        None
        if current_stats is None
        else round(percent_delta(recommended_stats["avg_accuracy"], current_stats["avg_accuracy"]), 1)
    )
    recommended_stats["cost_delta_pct"] = (
        None
        if current_stats is None
        else round(percent_delta(recommended_stats["median_cost"], current_stats["median_cost"]), 1)
    )
    recommended_stats["latency_delta_pct"] = (
        None
        if current_stats is None
        else round(percent_delta(recommended_stats["median_latency_ms"], current_stats["median_latency_ms"]), 1)
    )

    switch_recommended, policy_reason = should_switch(
        recommended_stats,
        current_stats,
        min_accuracy_gain=min_accuracy_gain,
        max_cost_increase_pct=max_cost_increase_pct,
        max_latency_increase_pct=max_latency_increase_pct,
    )
    final_suggestion_model = recommended_stats["model_id"] if switch_recommended else current_model
    warnings: List[str] = []
    if not current_model_found:
        warnings.append(
            f"Current model '{current_model}' was not found in the KNN neighbor set, so comparison deltas were skipped."
        )

    reason = build_reason(
        switch_recommended=switch_recommended,
        policy_reason=policy_reason,
        recommended_stats=recommended_stats,
        current_model=current_model,
        current_model_found=current_model_found,
        filter_level="semantic_knn",
    )

    result = {
        "complexity": complexity,
        "complexity_confidence": round(complexity_confidence, 3) if complexity_confidence is not None else None,
        "complexity_source": complexity_source,
        "quality_score": None,
        "use_case": use_case,
        "clarity": clarity,
        "clarity_source": clarity_source,
        "filter_level": "semantic_knn",
        "recommendation_mode": "semantic_best_value",
        "data_source": "knn",
        "current_model": current_model,
        "current_model_found": current_model_found,
        "current_model_stats": current_stats,
        "recommended_model": recommended_stats["model_id"],
        "recommended_provider": recommended_stats["provider"],
        "expected_accuracy": recommended_stats["avg_accuracy"],
        "expected_cost": recommended_stats["median_cost"],
        "expected_latency": recommended_stats["median_latency_ms"],
        "accuracy_delta": recommended_stats["accuracy_delta"],
        "accuracy_delta_pct": recommended_stats["accuracy_delta_pct"],
        "cost_delta_pct": recommended_stats["cost_delta_pct"],
        "latency_delta_pct": recommended_stats["latency_delta_pct"],
        "sample_size": recommended_stats["sample_count"],
        "slice_row_count": len(neighbors),
        "models_considered": len(ranked),
        "switch_recommended": switch_recommended,
        "final_suggestion_model": final_suggestion_model,
        "policy_reason": policy_reason,
        "reason": reason,
        "top_candidates": [build_knn_model_stats(row) for row in ranked[:5]],
        "warnings": warnings,
        "policy_thresholds": {
            "min_accuracy_gain": min_accuracy_gain,
            "max_cost_increase_pct": max_cost_increase_pct,
            "max_latency_increase_pct": max_latency_increase_pct,
        },
    }
    result["knn_neighbors_used"] = len(neighbors)
    result["embedding_cached"] = was_cached
    result["prompt_hash"] = prompt_hash
    result["knn_confidence"] = best.get("confidence")

    try:
        asyncio.create_task(_write_routing_log(result, request_id=str(uuid.uuid4())[:8]))
    except RuntimeError:
        pass

    return result


async def _write_routing_log(result: dict, request_id: str) -> None:
    try:
        from services.supabase_client import supabase

        supabase.table("routing_log").insert(
            {
                "request_id": request_id,
                "prompt_hash": result.get("prompt_hash"),
                "use_case": result.get("use_case"),
                "complexity": result.get("complexity"),
                "clarity": result.get("clarity"),
                "recommended_model": result.get("recommended_model"),
                "data_source": result.get("data_source"),
                "knn_neighbors": result.get("knn_neighbors_used"),
                "filter_level": result.get("filter_level"),
                "expected_accuracy": result.get("expected_accuracy"),
                "confidence": result.get("knn_confidence"),
            }
        ).execute()
    except Exception as exc:
        print(f"[ROUTING LOG ERROR] {exc}")


async def _shadow_knn_recommendation(prompt: str, use_case: str, slice_result: dict) -> None:
    """
    Runs semantic KNN routing in shadow mode and logs agreement with the slice
    recommender. It does not affect the user-facing recommendation response.
    """
    try:
        from services.embedding_service import get_or_compute_embedding
        from services.knn_search import aggregate_knn_signals, search_neighbors
        from services.supabase_client import supabase

        vector, _prompt_hash, was_cached = await get_or_compute_embedding(prompt, supabase)
        neighbors = search_neighbors(supabase, vector, use_case)

        if not neighbors:
            print(f"[SHADOW KNN] no_neighbors use_case={use_case} cached={was_cached}")
            return

        signals = aggregate_knn_signals(neighbors)
        if not signals:
            print(f"[SHADOW KNN] sparse_neighbors use_case={use_case} neighbors={len(neighbors)}")
            return

        knn_top = max(signals.values(), key=lambda item: item["sim_weighted_accuracy"])
        slice_top = slice_result.get("recommended_model", "?")
        agree = knn_top["model_id"] == slice_top
        print(
            f"[SHADOW KNN] knn={knn_top['model_id']} "
            f"acc={knn_top['sim_weighted_accuracy']} slice={slice_top} "
            f"agree={agree} neighbors={len(neighbors)} cached={was_cached}"
        )
    except Exception as exc:
        print(f"[SHADOW KNN ERROR] {exc}")


async def get_recommendation_options() -> dict:
    all_rows, data_source = await load_benchmark_rows_with_fallback(use_case=None)
    if not all_rows:
        return {
            "data_source": data_source,
            "use_cases": [],
            "models": [],
        }

    summary = summarize_models(all_rows)
    models = []
    for row in summary:
        use_cases = sorted({item["use_case"] for item in all_rows if item["model_id"] == row["model_id"]})
        models.append(
            {
                "model_id": row["model_id"],
                "provider": row["provider"],
                "avg_accuracy": round(float(row["avg_accuracy"]), 2),
                "median_cost": round(float(row["median_cost"]), 6),
                "median_latency_ms": round(float(row["median_latency_ms"]), 1),
                "sample_count": int(row["sample_count"]),
                "use_cases": use_cases,
            }
        )

    use_cases = [
        {
            "value": "text-generation",
            "label": "Text Generation",
            "description": "General writing, transformation, drafting, and conversational tasks.",
        },
        {
            "value": "code-generation",
            "label": "Code Generation",
            "description": "Implementation-heavy prompts, debugging, APIs, and engineering workflows.",
        },
        {
            "value": "reasoning",
            "label": "Reasoning",
            "description": "Multi-step logic, math, structured analysis, and careful problem solving.",
        },
    ]

    return {
        "data_source": data_source,
        "use_cases": use_cases,
        "models": models,
    }

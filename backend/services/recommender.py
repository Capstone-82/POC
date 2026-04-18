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

from services.supabase_client import get_benchmark_data, get_model_win_rates, get_prompt_logs

VALID_COMPLEXITIES = {"low", "mid", "high"}
VALID_CLARITIES = {"CLEAR", "PARTIAL", "UNCLEAR"}

MIN_SAMPLES_PER_MODEL = 5
ACCURACY_TOLERANCE = 2.0
MIN_WIN_RATE_ADVANTAGE = 0.10
MIN_COST_IMPROVEMENT_PCT = 15.0
MIN_LATENCY_IMPROVEMENT_PCT = 20.0
MIN_KNN_NEIGHBORS = 5

CLASSIFIER_PATH = (
    Path(__file__).resolve().parents[2] / "model_training" / "artifacts" / "classifier.pkl"
)
LOCAL_BENCHMARK_CSV = Path(__file__).resolve().parents[2] / "model_training" / "benchmark_results.csv"
LOCAL_PROMPT_LOGS_CSV = Path(__file__).resolve().parents[2] / "model_training" / "prompt_logs_rows.csv"

SCORE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "code-generation": {
        "win_rate": 0.25,
        "knn_accuracy": 0.25,
        "syntax_rate": 0.10,
        "cost": 0.20,
        "latency": 0.15,
        "confidence": 0.05,
    },
    "reasoning": {
        "win_rate": 0.25,
        "knn_accuracy": 0.25,
        "correctness": 0.15,
        "cost": 0.15,
        "latency": 0.15,
        "confidence": 0.05,
    },
    "text-generation": {
        "win_rate": 0.25,
        "knn_accuracy": 0.25,
        "cost": 0.20,
        "latency": 0.20,
        "confidence": 0.10,
    },
    "data-analysis": {
        "win_rate": 0.25,
        "knn_accuracy": 0.25,
        "cost": 0.25,
        "latency": 0.20,
        "confidence": 0.05,
    },
    "question-answering": {
        "win_rate": 0.25,
        "knn_accuracy": 0.25,
        "correctness": 0.15,
        "cost": 0.20,
        "latency": 0.10,
        "confidence": 0.05,
    },
    "_default": {
        "win_rate": 0.25,
        "knn_accuracy": 0.25,
        "cost": 0.25,
        "latency": 0.20,
        "confidence": 0.05,
    },
}

PAIRWISE_FALLBACK_CONFIDENCE = 0.35
PAIRWISE_MISSING_PENALTY = 0.85


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


def normalize_higher_better(value: float, min_value: float, max_value: float) -> float:
    if math.isclose(min_value, max_value):
        return 1.0
    return (value - min_value) / (max_value - min_value)


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
        "win_rate": (
            None if row.get("win_rate") is None else round(float(row["win_rate"]), 3)
        ),
        "syntax_pass_rate": (
            None if row.get("syntax_pass_rate") is None else round(float(row["syntax_pass_rate"]), 3)
        ),
        "correctness_rate": (
            None if row.get("correctness_rate") is None else round(float(row["correctness_rate"]), 3)
        ),
        "confidence": (
            None if row.get("confidence") is None else round(float(row["confidence"]), 3)
        ),
        "value_score": (
            None if row.get("value_score") is None else round(float(row["value_score"]), 4)
        ),
    }


def build_knn_model_stats(row: dict) -> dict:
    accuracy = float(
        row.get("sim_weighted_accuracy", row.get("fallback_accuracy", row.get("avg_accuracy", 0.0)))
    )
    return {
        "model_id": row["model_id"],
        "provider": row.get("provider", ""),
        "sample_count": int(row.get("sample_n", row.get("sample_count", 0))),
        "avg_accuracy": round(accuracy, 2),
        "median_accuracy": round(accuracy, 2),
        "median_cost": round(float(row.get("p50_cost", row.get("median_cost", 0.0))), 6),
        "median_latency_ms": round(float(row.get("p50_latency", row.get("median_latency_ms", 0.0))), 1),
        "win_rate": None if row.get("win_rate") is None else round(float(row["win_rate"]), 3),
        "syntax_pass_rate": (
            None if row.get("syntax_pass_rate") is None else round(float(row["syntax_pass_rate"]), 3)
        ),
        "correctness_rate": (
            None if row.get("correctness_rate") is None else round(float(row["correctness_rate"]), 3)
        ),
        "confidence": None if row.get("confidence") is None else round(float(row["confidence"]), 3),
        "value_score": None if row.get("value_score") is None else round(float(row["value_score"]), 4),
    }


def aggregate_knn_signals_v2(
    neighbors: List[dict],
    use_case: str,
    win_rates: Dict[str, Dict[str, Any]],
) -> Dict[str, dict]:
    grouped: Dict[str, List[dict]] = {}
    for row in neighbors:
        model_id = str(row.get("model_id", "")).strip()
        if model_id:
            grouped.setdefault(model_id, []).append(row)

    aggregated: Dict[str, dict] = {}
    for model_id, rows in grouped.items():
        from services.knn_search import MIN_MODEL_NEIGHBORS as _MIN_N
        if len(rows) < _MIN_N:
            continue

        sims = [float(row.get("similarity", 0.0)) for row in rows]
        costs = [float(row.get("cost", 0.0)) for row in rows]
        latencies = [float(row.get("latency_ms", 0.0)) for row in rows]
        accuracies = [
            float(row.get("avg_accuracy_score"))
            for row in rows
            if row.get("avg_accuracy_score") is not None
        ]
        total_sim = sum(sims)
        if total_sim <= 0:
            continue

        accuracy_pairs = [
            (float(row.get("similarity", 0.0)), float(row.get("avg_accuracy_score")))
            for row in rows
            if row.get("avg_accuracy_score") is not None
        ]
        weighted_accuracy_total = sum(sim for sim, _ in accuracy_pairs)
        fallback_accuracy = (
            sum(sim * acc for sim, acc in accuracy_pairs) / weighted_accuracy_total
            if weighted_accuracy_total > 0
            else 0.0
        )

        syntax_values = [
            1.0 if bool(row.get("syntax_pass")) else 0.0
            for row in rows
            if row.get("syntax_pass") is not None
        ]
        correctness_values = [
            1.0 if bool(row.get("is_correct")) else 0.0
            for row in rows
            if row.get("is_correct") is not None
        ]
        avg_similarity = sum(sims) / len(sims)
        sample_factor = min(len(rows) / 5.0, 1.0)

        pairwise_stats = win_rates.get(model_id, {})
        pairwise_win_rate = pairwise_stats.get("win_rate")
        pairwise_confidence = float(pairwise_stats.get("confidence", 0.0) or 0.0)

        aggregated[model_id] = {
            "model_id": model_id,
            "provider": rows[0].get("provider", ""),
            "fallback_accuracy": round(fallback_accuracy, 2),
            "sim_weighted_accuracy": round(fallback_accuracy, 2),
            "win_rate": pairwise_win_rate,
            "syntax_pass_rate": (
                round(sum(syntax_values) / len(syntax_values), 3) if syntax_values else None
            ),
            "correctness_rate": (
                round(sum(correctness_values) / len(correctness_values), 3) if correctness_values else None
            ),
            "p50_cost": round(float(median(costs)), 6),
            "p50_latency": round(float(median(latencies)), 1),
            "sample_n": len(rows),
            "avg_similarity": round(avg_similarity, 4),
            "pairwise_confidence": round(pairwise_confidence, 4),
            "decisive_matches": int(pairwise_stats.get("decisive_matches", 0) or 0),
            "tie_rate": float(pairwise_stats.get("tie_rate", 0.0) or 0.0),
            "confidence_signal": round(
                avg_similarity
                * sample_factor
                * (pairwise_confidence if pairwise_win_rate is not None else PAIRWISE_FALLBACK_CONFIDENCE),
                4,
            ),
        }

        if use_case == "code-generation" and aggregated[model_id]["syntax_pass_rate"] is not None:
            if float(aggregated[model_id]["syntax_pass_rate"]) < 0.85:
                aggregated[model_id]["confidence_signal"] *= 0.85

    return aggregated


def score_and_rank_knn_candidates(candidates: Dict[str, dict], use_case: str) -> List[dict]:
    ranked = [dict(item) for item in candidates.values()]
    if not ranked:
        return []

    weights = SCORE_WEIGHTS.get(use_case, SCORE_WEIGHTS["_default"])
    quality_values = []
    for item in ranked:
        if item.get("win_rate") is not None:
            quality_values.append(float(item["win_rate"]))
        else:
            fallback_quality = float(item.get("fallback_accuracy", item.get("sim_weighted_accuracy", 0.0))) / 100.0
            quality_values.append(fallback_quality * PAIRWISE_MISSING_PENALTY)

    # KNN accuracy signal — the whole point of semantic search
    accuracy_values = [
        float(item.get("sim_weighted_accuracy", item.get("fallback_accuracy", 0.0)))
        for item in ranked
    ]

    cost_values = [float(item["p50_cost"]) for item in ranked]
    latency_values = [float(item["p50_latency"]) for item in ranked]
    confidence_values = [float(item.get("confidence_signal", 0.0)) for item in ranked]
    syntax_values = [
        float(item["syntax_pass_rate"]) for item in ranked if item.get("syntax_pass_rate") is not None
    ]
    correctness_values = [
        float(item["correctness_rate"]) for item in ranked if item.get("correctness_rate") is not None
    ]

    quality_min, quality_max = min(quality_values), max(quality_values)
    accuracy_min, accuracy_max = min(accuracy_values), max(accuracy_values)
    cost_min, cost_max = min(cost_values), max(cost_values)
    latency_min, latency_max = min(latency_values), max(latency_values)
    confidence_min, confidence_max = min(confidence_values), max(confidence_values)
    syntax_min = min(syntax_values) if syntax_values else 0.0
    syntax_max = max(syntax_values) if syntax_values else 1.0
    correctness_min = min(correctness_values) if correctness_values else 0.0
    correctness_max = max(correctness_values) if correctness_values else 1.0

    for item, quality_signal, accuracy_signal in zip(ranked, quality_values, accuracy_values):
        quality_norm = normalize_higher_better(quality_signal, quality_min, quality_max)
        accuracy_norm = normalize_higher_better(accuracy_signal, accuracy_min, accuracy_max)
        cost_norm = normalize_lower_better(float(item["p50_cost"]), cost_min, cost_max)
        latency_norm = normalize_lower_better(float(item["p50_latency"]), latency_min, latency_max)
        confidence_norm = normalize_higher_better(
            float(item.get("confidence_signal", 0.0)),
            confidence_min,
            confidence_max,
        )
        syntax_norm = (
            normalize_higher_better(float(item["syntax_pass_rate"]), syntax_min, syntax_max)
            if item.get("syntax_pass_rate") is not None
            else 0.5
        )
        correctness_norm = (
            normalize_higher_better(float(item["correctness_rate"]), correctness_min, correctness_max)
            if item.get("correctness_rate") is not None
            else 0.5
        )

        if item.get("win_rate") is None:
            confidence_norm *= PAIRWISE_MISSING_PENALTY

        value_score = 0.0
        value_score += weights.get("win_rate", 0.0) * quality_norm
        value_score += weights.get("knn_accuracy", 0.0) * accuracy_norm
        value_score += weights.get("cost", 0.0) * cost_norm
        value_score += weights.get("latency", 0.0) * latency_norm
        value_score += weights.get("confidence", 0.0) * confidence_norm
        value_score += weights.get("syntax_rate", 0.0) * syntax_norm
        value_score += weights.get("correctness", 0.0) * correctness_norm

        item["confidence"] = round(confidence_norm, 3)
        item["value_score"] = round(value_score, 4)
        item["quality_signal"] = round(quality_signal, 4)

    ranked.sort(
        key=lambda item: (
            -item["value_score"],
            -float(item.get("quality_signal", 0.0)),
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
        return True, "No current model comparison was available, so the top-ranked candidate is recommended."

    win_rate_delta = recommended.get("win_rate_delta")
    cost_delta_pct = recommended.get("cost_delta_pct")
    latency_delta_pct = recommended.get("latency_delta_pct")

    if win_rate_delta is not None and win_rate_delta >= MIN_WIN_RATE_ADVANTAGE:
        return True, f"Win rate advantage is material at +{win_rate_delta:.1%}."

    if (
        cost_delta_pct is not None
        and cost_delta_pct <= -MIN_COST_IMPROVEMENT_PCT
        and (win_rate_delta is None or win_rate_delta >= -0.05)
    ):
        return True, f"Cost is lower by {abs(cost_delta_pct):.1f}% with comparable quality."

    if (
        latency_delta_pct is not None
        and latency_delta_pct <= -MIN_LATENCY_IMPROVEMENT_PCT
        and (win_rate_delta is None or win_rate_delta >= -0.05)
    ):
        return True, f"Latency is lower by {abs(latency_delta_pct):.1f}% with comparable quality."

    return False, "The recommended model is not materially better on win rate, cost, or latency."


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
    win_rates = await get_model_win_rates(use_case=use_case, complexity=complexity)
    if not win_rates:
        win_rates = await get_model_win_rates(use_case=use_case, complexity="all")

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

    slice_candidates: Dict[str, dict] = {}
    for row in summary:
        pairwise_stats = win_rates.get(row["model_id"], {})
        slice_candidates[row["model_id"]] = {
            "model_id": row["model_id"],
            "provider": row["provider"],
            "fallback_accuracy": row["avg_accuracy"],
            "sim_weighted_accuracy": row["avg_accuracy"],
            "win_rate": pairwise_stats.get("win_rate"),
            "syntax_pass_rate": None,
            "correctness_rate": None,
            "p50_cost": row["median_cost"],
            "p50_latency": row["median_latency_ms"],
            "sample_n": row["sample_count"],
            "pairwise_confidence": float(pairwise_stats.get("confidence", 0.0) or 0.0),
            "decisive_matches": int(pairwise_stats.get("decisive_matches", 0) or 0),
            "tie_rate": float(pairwise_stats.get("tie_rate", 0.0) or 0.0),
            "confidence_signal": min(row["sample_count"] / 10.0, 1.0)
            * (
                float(pairwise_stats.get("confidence", 0.0) or 0.0)
                if pairwise_stats.get("win_rate") is not None
                else PAIRWISE_FALLBACK_CONFIDENCE
            ),
        }

    ranked_summary = score_and_rank_knn_candidates(slice_candidates, use_case=use_case)
    best_row = ranked_summary[0]
    recommended_stats = build_knn_model_stats(best_row)

    current_row = next((row for row in ranked_summary if row["model_id"] == current_model), None)
    current_stats = build_knn_model_stats(current_row) if current_row else None
    current_model_found = current_stats is not None

    recommended_stats["win_rate_delta"] = (
        None
        if current_stats is None or recommended_stats.get("win_rate") is None or current_stats.get("win_rate") is None
        else round(recommended_stats["win_rate"] - current_stats["win_rate"], 3)
    )
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
        "expected_win_rate": recommended_stats.get("win_rate"),
        "expected_syntax_pass_rate": recommended_stats.get("syntax_pass_rate"),
        "expected_correctness_rate": recommended_stats.get("correctness_rate"),
        "win_rate_delta": recommended_stats.get("win_rate_delta"),
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
        "top_candidates": [build_knn_model_stats(row) for row in ranked_summary[:5]],
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
        search_neighbors,
    )
    from services.model_registry import get_model_ids_for_use_case
    from services.supabase_client import supabase

    vector, prompt_hash, was_cached = await get_or_compute_embedding(prompt, supabase)

    # Fix 3: fetch win_rates for specific complexity, fall back to 'all' if empty
    win_rates = await get_model_win_rates(use_case=use_case, complexity=complexity)
    if not win_rates:
        win_rates = await get_model_win_rates(use_case=use_case, complexity="all")
    print(f"[KNN] win_rates loaded: {len(win_rates)} models for {use_case}/{complexity}")

    neighbors = search_neighbors(supabase, vector, use_case)
    # Fix 1: never hard-raise on low neighbor count — widen search progressively
    if len(neighbors) < MIN_KNN_NEIGHBORS:
        neighbors = search_neighbors(
            supabase,
            vector,
            use_case,
            k=FALLBACK_K,
            min_similarity=FALLBACK_SIMILARITY,
        )
    if len(neighbors) < MIN_KNN_NEIGHBORS:
        # Last resort: maximum k, minimum similarity threshold
        neighbors = search_neighbors(
            supabase,
            vector,
            use_case,
            k=AGGREGATION_FALLBACK_K,
            min_similarity=0.0,
        )

    allowed_models = get_model_ids_for_use_case(use_case)
    def aggregate_allowed(rows: List[dict]) -> Dict[str, dict]:
        return {
            model_id: signals
            for model_id, signals in aggregate_knn_signals_v2(rows, use_case=use_case, win_rates=win_rates).items()
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

    ranked = score_and_rank_knn_candidates(knn_signals, use_case=use_case)
    # Fix 5: don't raise — explicitly fall back to slice with informative warning
    if not ranked:
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
            fallback_warning="KNN neighbors were too sparse after aggregation; slice fallback used.",
        )

    best = ranked[0]
    recommended_stats = build_knn_model_stats(best)
    current_row = next((row for row in ranked if row["model_id"] == current_model), None)
    current_stats = build_knn_model_stats(current_row) if current_row else None
    current_model_found = current_stats is not None

    recommended_stats["win_rate_delta"] = (
        None
        if current_stats is None or recommended_stats.get("win_rate") is None or current_stats.get("win_rate") is None
        else round(recommended_stats["win_rate"] - current_stats["win_rate"], 3)
    )
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
        "expected_win_rate": recommended_stats.get("win_rate"),
        "expected_syntax_pass_rate": recommended_stats.get("syntax_pass_rate"),
        "expected_correctness_rate": recommended_stats.get("correctness_rate"),
        "win_rate_delta": recommended_stats.get("win_rate_delta"),
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

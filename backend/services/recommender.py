from __future__ import annotations

import math
import pickle
import re
import csv
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

CLASSIFIER_PATH = (
    Path(__file__).resolve().parents[2] / "model_training" / "artifacts" / "classifier.pkl"
)
LOCAL_BENCHMARK_CSV = Path(__file__).resolve().parents[2] / "model_training" / "benchmark_results.csv"
LOCAL_PROMPT_LOGS_CSV = Path(__file__).resolve().parents[2] / "model_training" / "prompt_logs_rows.csv"


def normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip().lower())


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


def infer_complexity(prompt: str, classifier: Optional[Any]) -> Tuple[str, Optional[float], str]:
    if classifier is not None:
        prediction = str(classifier.predict([prompt])[0]).strip().lower()
        confidence = None
        if hasattr(classifier, "predict_proba"):
            try:
                confidence = float(max(classifier.predict_proba([prompt])[0]))
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
            model_id = str(row["model_id"]).strip()
            provider = str(row.get("provider", "")).strip()
            use_case = str(row["use_case"]).strip().lower()
            complexity = str(row["prompt_complexity"]).strip().lower()
            clarity = str(row["clarity"]).strip().upper()
            accuracy_score = float(row["accuracy_score"])
            cost = float(row["cost"])
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


def should_switch(recommended: dict, current: Optional[dict]) -> Tuple[bool, str]:
    if current is None:
        return True, "No current model comparison was available, so this is the best value option in the matched benchmark slice."

    accuracy_gain = recommended["accuracy_delta"]
    cost_delta_pct = recommended["cost_delta_pct"]
    latency_delta_pct = recommended["latency_delta_pct"]

    if accuracy_gain is not None and accuracy_gain >= MIN_ACCURACY_GAIN:
        return True, "Accuracy gain is large enough to justify switching."

    if (
        cost_delta_pct is not None
        and cost_delta_pct <= -MIN_COST_IMPROVEMENT_PCT
        and (accuracy_gain is None or accuracy_gain >= -1.0)
    ):
        return True, "Cost drops meaningfully without a material quality loss."

    if (
        latency_delta_pct is not None
        and latency_delta_pct <= -MIN_LATENCY_IMPROVEMENT_PCT
        and (accuracy_gain is None or accuracy_gain >= -1.0)
    ):
        return True, "Latency improves meaningfully without a material quality loss."

    return False, "Current model is already close enough that switching would mostly be noise."


def build_reason(
    switch_recommended: bool,
    policy_reason: str,
    recommended_stats: dict,
    current_model: str,
    current_model_found: bool,
    filter_level: str,
) -> str:
    recommended_name = f"{recommended_stats['provider']}/{recommended_stats['model_id']}"
    if not current_model_found:
        return (
            f"{recommended_name} is the best value option in the {filter_level} benchmark slice. "
            f"{policy_reason}"
        )

    if switch_recommended:
        return (
            f"Switch from {current_model} to {recommended_stats['model_id']}. "
            f"{policy_reason}"
        )

    return (
        f"Stay on {current_model}. {policy_reason} "
        f"The best alternative in the {filter_level} slice was {recommended_name}."
    )


async def get_recommendation(use_case: str, prompt: str, current_model: str) -> dict:
    classifier = load_complexity_classifier()
    complexity, complexity_confidence, complexity_source = infer_complexity(prompt, classifier)
    clarity, clarity_source = await infer_clarity(prompt, use_case)

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

    switch_recommended, policy_reason = should_switch(recommended_stats, current_stats)
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
    if not current_model_found:
        warnings.append(
            f"Current model '{current_model}' was not found in the matched benchmark slice, so comparison deltas were skipped."
        )

    return {
        "complexity": complexity,
        "complexity_confidence": round(complexity_confidence, 3) if complexity_confidence is not None else None,
        "complexity_source": complexity_source,
        "quality_score": None,
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
    }


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

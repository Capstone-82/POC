"""
Benchmark-driven model recommender prototype.

This version is intentionally standalone so we can test the decision policy
before wiring it into the backend and frontend.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BENCHMARK_CSV = BASE_DIR / "benchmark_results.csv"
PROMPT_LOGS_CSV = BASE_DIR / "prompt_logs_rows.csv"
CLASSIFIER_PATH = BASE_DIR / "artifacts" / "classifier.pkl"

VALID_USE_CASES = {"text-generation", "code-generation", "reasoning"}
VALID_COMPLEXITIES = {"low", "mid", "high"}
VALID_CLARITIES = {"CLEAR", "PARTIAL", "UNCLEAR"}

MIN_SAMPLES_PER_MODEL = 5
ACCURACY_TOLERANCE = 2.0
MIN_ACCURACY_GAIN = 2.0
MIN_COST_IMPROVEMENT_PCT = 15.0
MIN_LATENCY_IMPROVEMENT_PCT = 20.0


@dataclass
class PromptSignals:
    use_case: str
    complexity: str
    complexity_confidence: float | None
    clarity: str
    clarity_source: str


def load_classifier_training_rows() -> pd.DataFrame:
    df = pd.read_csv(BENCHMARK_CSV, usecols=["prompt", "prompt_complexity"])
    df = df.dropna(subset=["prompt", "prompt_complexity"]).copy()
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df["prompt_complexity"] = df["prompt_complexity"].astype(str).str.strip().str.lower()
    df = df[df["prompt"] != ""]
    df = df[df["prompt_complexity"].isin(VALID_COMPLEXITIES)]
    df = df.drop_duplicates(subset=["prompt", "prompt_complexity"]).reset_index(drop=True)
    return df


def load_benchmark_rows() -> pd.DataFrame:
    df = pd.read_csv(BENCHMARK_CSV)
    df = df.dropna(
        subset=[
            "model_id",
            "use_case",
            "prompt_complexity",
            "clarity",
            "accuracy_score",
            "cost",
            "latency_ms",
        ]
    ).copy()
    df["use_case"] = df["use_case"].astype(str).str.strip().str.lower()
    df["prompt_complexity"] = df["prompt_complexity"].astype(str).str.strip().str.lower()
    df["clarity"] = df["clarity"].astype(str).str.strip().str.upper()
    df["model_id"] = df["model_id"].astype(str).str.strip()
    df["provider"] = df["provider"].fillna("").astype(str).str.strip()
    df["accuracy_score"] = pd.to_numeric(df["accuracy_score"], errors="coerce")
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    df = df.dropna(subset=["accuracy_score", "cost", "latency_ms"])
    return df


def load_prompt_logs() -> pd.DataFrame:
    df = pd.read_csv(PROMPT_LOGS_CSV)
    df = df.dropna(subset=["prompt", "use_case", "clarity"]).copy()
    df["prompt"] = df["prompt"].astype(str)
    df["normalized_prompt"] = df["prompt"].map(normalize_prompt)
    df["use_case"] = df["use_case"].astype(str).str.strip().str.lower()
    df["clarity"] = df["clarity"].astype(str).str.strip().str.upper()
    return df


def load_complexity_classifier():
    if not CLASSIFIER_PATH.exists():
        return None
    try:
        with CLASSIFIER_PATH.open("rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError:
        return None


def train_complexity_classifier() -> dict[str, Any]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import Pipeline
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Training requires scikit-learn. Install it before running --train-classifier."
        ) from exc

    df = load_classifier_training_rows()
    if df.empty:
        raise RuntimeError("No labeled prompt complexity rows were found in benchmark_results.csv.")

    X = df["prompt"].tolist()
    y = df["prompt_complexity"].tolist()

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=5.0,
                    max_iter=1000,
                    solver="lbfgs",
                    multi_class="multinomial",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    min_class_count = int(df["prompt_complexity"].value_counts().min())
    cv_folds = min(5, min_class_count)
    cv_accuracy_mean = None
    cv_accuracy_std = None
    if cv_folds >= 2:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy")
        cv_accuracy_mean = float(cv_scores.mean())
        cv_accuracy_std = float(cv_scores.std())

    pipeline.fit(X, y)

    CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CLASSIFIER_PATH.open("wb") as f:
        pickle.dump(pipeline, f)

    return {
        "artifact_path": str(CLASSIFIER_PATH),
        "training_rows": int(len(df)),
        "class_distribution": {
            key: int(value)
            for key, value in df["prompt_complexity"].value_counts().sort_index().items()
        },
        "cv_folds": cv_folds,
        "cv_accuracy_mean": cv_accuracy_mean,
        "cv_accuracy_std": cv_accuracy_std,
        "artifact_size_kb": round(CLASSIFIER_PATH.stat().st_size / 1024.0, 1),
    }


def normalize_prompt(prompt: str) -> str:
    prompt = re.sub(r"\s+", " ", prompt.strip().lower())
    return prompt


def infer_complexity(prompt: str, classifier: Any | None) -> tuple[str, float | None, str]:
    if classifier is not None:
        prediction = classifier.predict([prompt])[0]
        confidence = None
        if hasattr(classifier, "predict_proba"):
            try:
                confidence = float(max(classifier.predict_proba([prompt])[0]))
            except Exception:
                confidence = None
        prediction = str(prediction).strip().lower()
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


def infer_clarity(prompt: str, use_case: str, prompt_logs: pd.DataFrame) -> tuple[str, str]:
    normalized = normalize_prompt(prompt)
    exact = prompt_logs[
        (prompt_logs["use_case"] == use_case) & (prompt_logs["normalized_prompt"] == normalized)
    ]
    if not exact.empty:
        top_clarity = (
            exact["clarity"].value_counts().sort_values(ascending=False).index.tolist()[0]
        )
        return top_clarity, "prompt_logs_exact"

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


def infer_signals(
    prompt: str,
    use_case: str,
    classifier: Any | None,
    prompt_logs: pd.DataFrame,
    complexity_override: str | None = None,
    clarity_override: str | None = None,
) -> PromptSignals:
    if complexity_override:
        complexity = complexity_override
        complexity_confidence = None
    else:
        complexity, complexity_confidence, _ = infer_complexity(prompt, classifier)

    if clarity_override:
        clarity = clarity_override
        clarity_source = "manual_override"
    else:
        clarity, clarity_source = infer_clarity(prompt, use_case, prompt_logs)
    return PromptSignals(
        use_case=use_case,
        complexity=complexity,
        complexity_confidence=complexity_confidence,
        clarity=clarity,
        clarity_source=clarity_source,
    )


def filter_candidates(df: pd.DataFrame, signals: PromptSignals) -> tuple[pd.DataFrame, str]:
    filter_tiers = [
        (
            "exact",
            (df["use_case"] == signals.use_case)
            & (df["prompt_complexity"] == signals.complexity)
            & (df["clarity"] == signals.clarity),
        ),
        (
            "use_case_plus_complexity",
            (df["use_case"] == signals.use_case)
            & (df["prompt_complexity"] == signals.complexity),
        ),
        ("use_case_only", df["use_case"] == signals.use_case),
    ]

    for tier_name, mask in filter_tiers:
        subset = df[mask].copy()
        if subset.empty:
            continue
        model_support = subset.groupby("model_id").size()
        supported_models = model_support[model_support >= MIN_SAMPLES_PER_MODEL].index
        supported_subset = subset[subset["model_id"].isin(supported_models)].copy()
        if not supported_subset.empty:
            return supported_subset, tier_name

    return pd.DataFrame(), "none"


def build_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["model_id", "provider"], dropna=False)
        .agg(
            sample_count=("accuracy_score", "count"),
            avg_accuracy=("accuracy_score", "mean"),
            median_accuracy=("accuracy_score", "median"),
            median_cost=("cost", "median"),
            median_latency_ms=("latency_ms", "median"),
            avg_cost=("cost", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(
        ["avg_accuracy", "sample_count", "median_cost", "median_latency_ms"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def pick_best_value_model(summary: pd.DataFrame) -> pd.Series:
    top_accuracy = float(summary["avg_accuracy"].max())
    shortlist = summary[summary["avg_accuracy"] >= top_accuracy - ACCURACY_TOLERANCE].copy()

    cost_min = float(shortlist["median_cost"].min())
    cost_max = float(shortlist["median_cost"].max())
    latency_min = float(shortlist["median_latency_ms"].min())
    latency_max = float(shortlist["median_latency_ms"].max())

    def normalize_lower_better(value: float, min_value: float, max_value: float) -> float:
        if math.isclose(min_value, max_value):
            return 1.0
        return 1.0 - ((value - min_value) / (max_value - min_value))

    shortlist["value_score"] = shortlist.apply(
        lambda row: (
            0.75 * normalize_lower_better(float(row["median_cost"]), cost_min, cost_max)
            + 0.25
            * normalize_lower_better(
                float(row["median_latency_ms"]), latency_min, latency_max
            )
        ),
        axis=1,
    )
    shortlist = shortlist.sort_values(
        ["value_score", "avg_accuracy", "sample_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return shortlist.iloc[0]


def percent_delta(new_value: float, old_value: float) -> float | None:
    if old_value is None or old_value == 0:
        return None
    return ((new_value - old_value) / old_value) * 100.0


def build_current_model_stats(
    summary: pd.DataFrame, current_model: str | None
) -> dict[str, Any] | None:
    if not current_model:
        return None
    current_rows = summary[summary["model_id"] == current_model]
    if current_rows.empty:
        return None
    row = current_rows.iloc[0]
    return {
        "model_id": row["model_id"],
        "provider": row["provider"],
        "sample_count": int(row["sample_count"]),
        "avg_accuracy": round(float(row["avg_accuracy"]), 2),
        "median_cost": round(float(row["median_cost"]), 6),
        "median_latency_ms": round(float(row["median_latency_ms"]), 1),
    }


def should_switch(
    recommended: dict[str, Any],
    current: dict[str, Any] | None,
) -> tuple[bool, str]:
    if current is None:
        return True, "No current model was supplied, so this is the best value option in the matched benchmark slice."

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


def recommend(
    prompt: str,
    use_case: str,
    current_model: str | None = None,
    complexity_override: str | None = None,
    clarity_override: str | None = None,
    classifier: Any | None = None,
    benchmark_df: pd.DataFrame | None = None,
    prompt_logs: pd.DataFrame | None = None,
) -> dict[str, Any]:
    use_case = use_case.strip().lower()
    if use_case not in VALID_USE_CASES:
        raise ValueError(f"Unsupported use_case '{use_case}'. Expected one of {sorted(VALID_USE_CASES)}.")
    if complexity_override:
        complexity_override = complexity_override.strip().lower()
        if complexity_override not in VALID_COMPLEXITIES:
            raise ValueError(
                f"Unsupported complexity '{complexity_override}'. Expected one of {sorted(VALID_COMPLEXITIES)}."
            )
    if clarity_override:
        clarity_override = clarity_override.strip().upper()
        if clarity_override not in VALID_CLARITIES:
            raise ValueError(
                f"Unsupported clarity '{clarity_override}'. Expected one of {sorted(VALID_CLARITIES)}."
            )

    benchmark_df = benchmark_df if benchmark_df is not None else load_benchmark_rows()
    prompt_logs = prompt_logs if prompt_logs is not None else load_prompt_logs()
    classifier = classifier if classifier is not None else load_complexity_classifier()

    signals = infer_signals(
        prompt,
        use_case,
        classifier,
        prompt_logs,
        complexity_override=complexity_override,
        clarity_override=clarity_override,
    )
    candidate_rows, filter_level = filter_candidates(benchmark_df, signals)

    if candidate_rows.empty:
        return {
            "prompt": prompt,
            "signals": signals.__dict__,
            "error": "No sufficiently supported benchmark slice was found for this prompt.",
        }

    summary = build_model_summary(candidate_rows)
    best_row = pick_best_value_model(summary)
    current_stats = build_current_model_stats(summary, current_model)

    recommended_stats = {
        "model_id": best_row["model_id"],
        "provider": best_row["provider"],
        "sample_count": int(best_row["sample_count"]),
        "avg_accuracy": round(float(best_row["avg_accuracy"]), 2),
        "median_accuracy": round(float(best_row["median_accuracy"]), 2),
        "median_cost": round(float(best_row["median_cost"]), 6),
        "median_latency_ms": round(float(best_row["median_latency_ms"]), 1),
    }

    if current_stats is not None:
        recommended_stats["accuracy_delta"] = round(
            recommended_stats["avg_accuracy"] - current_stats["avg_accuracy"], 2
        )
        recommended_stats["cost_delta_pct"] = (
            None
            if current_stats["median_cost"] == 0
            else round(
                percent_delta(
                    recommended_stats["median_cost"], current_stats["median_cost"]
                ),
                1,
            )
        )
        recommended_stats["latency_delta_pct"] = (
            None
            if current_stats["median_latency_ms"] == 0
            else round(
                percent_delta(
                    recommended_stats["median_latency_ms"],
                    current_stats["median_latency_ms"],
                ),
                1,
            )
        )
    else:
        recommended_stats["accuracy_delta"] = None
        recommended_stats["cost_delta_pct"] = None
        recommended_stats["latency_delta_pct"] = None

    switch_recommended, policy_reason = should_switch(recommended_stats, current_stats)
    final_model = recommended_stats["model_id"] if switch_recommended else (current_model or recommended_stats["model_id"])

    top_candidates = []
    for _, row in summary.head(5).iterrows():
        top_candidates.append(
            {
                "model_id": row["model_id"],
                "provider": row["provider"],
                "sample_count": int(row["sample_count"]),
                "avg_accuracy": round(float(row["avg_accuracy"]), 2),
                "median_cost": round(float(row["median_cost"]), 6),
                "median_latency_ms": round(float(row["median_latency_ms"]), 1),
            }
        )

    return {
        "prompt": prompt,
        "signals": signals.__dict__,
        "filter_level": filter_level,
        "slice_row_count": int(len(candidate_rows)),
        "models_considered": int(summary["model_id"].nunique()),
        "current_model": current_stats,
        "recommended_model": recommended_stats,
        "final_suggestion_model": final_model,
        "switch_recommended": switch_recommended,
        "policy_reason": policy_reason,
        "top_candidates": top_candidates,
    }


def format_result(result: dict[str, Any]) -> str:
    if "error" in result:
        return json.dumps(result, indent=2)

    signals = result["signals"]
    lines = [
        "=" * 88,
        "MODEL RECOMMENDATION V2",
        "=" * 88,
        f"Use case:           {signals['use_case']}",
        f"Prompt complexity:  {signals['complexity']}"
        + (
            f" (confidence {signals['complexity_confidence']:.1%})"
            if signals["complexity_confidence"] is not None
            else ""
        ),
        f"Clarity:            {signals['clarity']} ({signals['clarity_source']})",
        f"Filter level:       {result['filter_level']}",
        f"Slice rows:         {result['slice_row_count']}",
        f"Models considered:  {result['models_considered']}",
        "-" * 88,
    ]

    recommended = result["recommended_model"]
    lines.append(
        f"Recommended model:  {recommended['provider']}/{recommended['model_id']}"
    )
    lines.append(
        f"Expected metrics:   accuracy={recommended['avg_accuracy']:.2f}  "
        f"cost=${recommended['median_cost']:.6f}  latency={recommended['median_latency_ms']:.0f}ms  "
        f"(n={recommended['sample_count']})"
    )

    current = result.get("current_model")
    if current:
        lines.append(
            f"Current model:      {current['provider']}/{current['model_id']}  "
            f"accuracy={current['avg_accuracy']:.2f}  cost=${current['median_cost']:.6f}  "
            f"latency={current['median_latency_ms']:.0f}ms"
        )
        lines.append(
            "Delta vs current:   "
            f"accuracy={recommended['accuracy_delta']:+.2f}  "
            f"cost={recommended['cost_delta_pct']:+.1f}%  "
            f"latency={recommended['latency_delta_pct']:+.1f}%"
        )

    lines.append(
        f"Switch?             {'YES' if result['switch_recommended'] else 'NO'}"
    )
    lines.append(f"Policy reason:      {result['policy_reason']}")
    lines.append("-" * 88)
    lines.append("Top candidates:")
    for idx, candidate in enumerate(result["top_candidates"], start=1):
        lines.append(
            f"  {idx}. {candidate['provider']}/{candidate['model_id']}  "
            f"acc={candidate['avg_accuracy']:.2f}  "
            f"cost=${candidate['median_cost']:.6f}  "
            f"lat={candidate['median_latency_ms']:.0f}ms  "
            f"n={candidate['sample_count']}"
        )
    lines.append("=" * 88)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark-driven model recommendation prototype.")
    parser.add_argument(
        "--train-classifier",
        action="store_true",
        help="Train and save the prompt complexity classifier from benchmark_results.csv.",
    )
    parser.add_argument("--prompt", help="User prompt to score.")
    parser.add_argument(
        "--use-case",
        choices=sorted(VALID_USE_CASES),
        help="Use case supplied by the user.",
    )
    parser.add_argument("--current-model", help="Current model to compare against.")
    parser.add_argument(
        "--complexity",
        choices=sorted(VALID_COMPLEXITIES),
        help="Optional override for prompt complexity while testing.",
    )
    parser.add_argument(
        "--clarity",
        choices=sorted(VALID_CLARITIES),
        help="Optional override for prompt clarity while testing.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON instead of the formatted report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_classifier:
        result = train_complexity_classifier()
        print(json.dumps(result, indent=2))
        return
    if not args.prompt or not args.use_case:
        raise SystemExit("--prompt and --use-case are required unless --train-classifier is used.")
    result = recommend(
        prompt=args.prompt,
        use_case=args.use_case,
        current_model=args.current_model,
        complexity_override=args.complexity,
        clarity_override=args.clarity,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_result(result))


if __name__ == "__main__":
    main()

"""
LLM Model Recommendation System
================================
Pipeline:
  1. Lightweight Prompt Classifier (TF-IDF + Logistic Regression)
     - Trained on merged_dataset.csv
  2. Model Performance Profiles (pre-computed from benchmark CSVs)
  3. Recommendation Engine (delta-based comparisons)

Usage:
  python recommend.py                         # Interactive mode
  python recommend.py --train                  # Train classifier + build profiles
  python recommend.py --prompt "your prompt"   # Single prediction
"""

import os
import re
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CLASSIFIER_PATH = BASE_DIR / "artifacts" / "classifier.pkl"
PROFILES_PATH = BASE_DIR / "artifacts" / "model_profiles.json"
MERGED_CSV = BASE_DIR / "merged_dataset.csv"
PHASE1_CSV = BASE_DIR / "phase-1-dataset.csv"
PHASE1_1_CSV = BASE_DIR / "phase-1.1-dataset.csv"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Lightweight Prompt Classifier
# ═══════════════════════════════════════════════════════════════════════════════

def train_classifier():
    """Train a TF-IDF + Logistic Regression classifier on merged_dataset.csv"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report, confusion_matrix

    print("=" * 60)
    print("STEP 1: Training Lightweight Prompt Classifier")
    print("=" * 60)

    # Load data
    df = pd.read_csv(MERGED_CSV)
    df = df.dropna(subset=["prompt", "complexity"])
    df["complexity"] = df["complexity"].str.strip().str.lower()

    print(f"\nDataset: {len(df)} prompts")
    print(f"Class distribution:\n{df['complexity'].value_counts().to_string()}")

    X = df["prompt"].values
    y = df["complexity"].values

    # Build pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),       # unigrams + bigrams
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,        # log-scaled TF
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w+\b"
        )),
        ("clf", LogisticRegression(
            C=5.0,
            max_iter=1000,
            solver="lbfgs",
            multi_class="multinomial",
            class_weight="balanced",
            random_state=42
        ))
    ])

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy")
    print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Per-fold: {[f'{s:.4f}' for s in cv_scores]}")

    # Train on full data
    pipeline.fit(X, y)

    # Evaluation on training set (for comparison with notebook)
    y_pred = pipeline.predict(X)
    print(f"\nTraining Set Classification Report:")
    print(classification_report(y, y_pred))
    print(f"Confusion Matrix:\n{confusion_matrix(y, y_pred)}")

    # Save
    os.makedirs(CLASSIFIER_PATH.parent, exist_ok=True)
    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n✅ Classifier saved to: {CLASSIFIER_PATH}")
    print(f"   Model size: {CLASSIFIER_PATH.stat().st_size / 1024:.1f} KB")

    return pipeline


def load_classifier():
    """Load the trained classifier"""
    with open(CLASSIFIER_PATH, "rb") as f:
        return pickle.load(f)


def classify_prompt(pipeline, prompt: str) -> dict:
    """Classify a prompt and return class + probabilities"""
    prediction = pipeline.predict([prompt])[0]
    probas = pipeline.predict_proba([prompt])[0]
    classes = pipeline.classes_

    return {
        "complexity": prediction,
        "confidence": float(max(probas)),
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probas)}
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Build Model Performance Profiles
# ═══════════════════════════════════════════════════════════════════════════════

def build_model_profiles():
    """
    Aggregate benchmark data to build per-model, per-complexity performance profiles.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Building Model Performance Profiles")
    print("=" * 60)

    # Load and combine benchmark datasets
    dfs = []
    for csv_path in [PHASE1_CSV, PHASE1_1_CSV]:
        if csv_path.exists():
            print(f"  Loading {csv_path.name}...")
            df = pd.read_csv(csv_path, low_memory=False)
            dfs.append(df)
            print(f"    → {len(df)} rows")

    if not dfs:
        print("❌ No benchmark CSVs found!")
        return {}

    df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(df)} benchmark records")

    # Clean
    df["prompt_complexity"] = df["prompt_complexity"].str.strip().str.lower()
    df["accuracy_score"] = pd.to_numeric(df["accuracy_score"], errors="coerce")
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    df["tokens"] = pd.to_numeric(df["tokens"], errors="coerce")

    # Drop rows with missing critical fields
    df = df.dropna(subset=["model_id", "prompt_complexity", "accuracy_score"])

    print(f"After cleaning: {len(df)} records")
    print(f"Unique models: {df['model_id'].nunique()}")
    print(f"Unique providers: {df['provider'].nunique()}")

    # Aggregate per model + complexity
    agg = df.groupby(["provider", "model_id", "prompt_complexity"]).agg(
        avg_accuracy=("accuracy_score", "mean"),
        avg_cost=("cost", "mean"),
        avg_latency_ms=("latency_ms", "mean"),
        avg_tokens=("tokens", "mean"),
        sample_count=("accuracy_score", "count"),
        median_accuracy=("accuracy_score", "median"),
        std_accuracy=("accuracy_score", "std"),
    ).reset_index()

    # Also build an overall profile per model (across all complexity levels)
    overall = df.groupby(["provider", "model_id"]).agg(
        overall_avg_accuracy=("accuracy_score", "mean"),
        overall_avg_cost=("cost", "mean"),
        overall_avg_latency_ms=("latency_ms", "mean"),
        overall_sample_count=("accuracy_score", "count"),
    ).reset_index()

    # Structure into nested dict
    profiles = {}
    for _, row in agg.iterrows():
        model_key = row["model_id"]
        complexity = row["prompt_complexity"]

        if model_key not in profiles:
            # Get overall stats
            ov = overall[overall["model_id"] == model_key].iloc[0] if len(overall[overall["model_id"] == model_key]) > 0 else None
            profiles[model_key] = {
                "provider": row["provider"],
                "model_id": model_key,
                "overall": {
                    "avg_accuracy": round(float(ov["overall_avg_accuracy"]), 2) if ov is not None else 0,
                    "avg_cost": round(float(ov["overall_avg_cost"]), 6) if ov is not None else 0,
                    "avg_latency_ms": round(float(ov["overall_avg_latency_ms"]), 1) if ov is not None else 0,
                    "sample_count": int(ov["overall_sample_count"]) if ov is not None else 0,
                },
                "by_complexity": {}
            }

        profiles[model_key]["by_complexity"][complexity] = {
            "avg_accuracy": round(float(row["avg_accuracy"]), 2),
            "avg_cost": round(float(row["avg_cost"]), 6),
            "avg_latency_ms": round(float(row["avg_latency_ms"]), 1),
            "avg_tokens": round(float(row["avg_tokens"]), 1),
            "sample_count": int(row["sample_count"]),
            "median_accuracy": round(float(row["median_accuracy"]), 2),
            "std_accuracy": round(float(row["std_accuracy"]), 2) if not pd.isna(row["std_accuracy"]) else 0,
        }

    # Save
    os.makedirs(PROFILES_PATH.parent, exist_ok=True)
    with open(PROFILES_PATH, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"\n✅ Model profiles saved: {len(profiles)} models")
    print(f"   File: {PROFILES_PATH}")

    # Print top models per complexity
    for complexity in ["low", "mid", "high"]:
        subset = agg[agg["prompt_complexity"] == complexity].sort_values("avg_accuracy", ascending=False)
        print(f"\n  Top 5 models for '{complexity}' prompts:")
        for _, r in subset.head(5).iterrows():
            print(f"    {r['provider']:>12s}/{r['model_id']:<25s}  "
                  f"acc={r['avg_accuracy']:.1f}  cost=${r['avg_cost']:.6f}  "
                  f"lat={r['avg_latency_ms']:.0f}ms  (n={r['sample_count']})")

    return profiles


def load_profiles():
    """Load pre-computed model profiles"""
    with open(PROFILES_PATH, "r") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Recommendation Engine
# ═══════════════════════════════════════════════════════════════════════════════

def recommend(prompt: str, current_model: str = None, pipeline=None, profiles=None,
              top_n: int = 3, weights: dict = None):
    """
    Given a user prompt, classify it and recommend the best model(s).

    Args:
        prompt: The user's prompt text
        current_model: The model currently being used (for delta comparison)
        pipeline: Trained classifier pipeline
        profiles: Model performance profiles dict
        top_n: Number of recommendations to return
        weights: Custom scoring weights (default: accuracy-heavy)

    Returns:
        dict with classification, recommendations, and deltas
    """
    if pipeline is None:
        pipeline = load_classifier()
    if profiles is None:
        profiles = load_profiles()
    if weights is None:
        weights = {"accuracy": 0.50, "cost": 0.25, "latency": 0.25}

    # 1. Classify the prompt
    classification = classify_prompt(pipeline, prompt)
    complexity = classification["complexity"]

    # 2. Get all models' performance for this complexity
    candidates = []
    for model_key, profile in profiles.items():
        if complexity in profile.get("by_complexity", {}):
            stats = profile["by_complexity"][complexity]
            if stats["sample_count"] >= 3:  # Minimum samples threshold
                candidates.append({
                    "model_id": model_key,
                    "provider": profile["provider"],
                    **stats
                })

    if not candidates:
        return {
            "classification": classification,
            "error": f"No model data available for '{complexity}' complexity"
        }

    # 3. Score each model (normalized composite)
    df_candidates = pd.DataFrame(candidates)

    # Normalize each metric to [0, 1]
    acc_range = df_candidates["avg_accuracy"].max() - df_candidates["avg_accuracy"].min()
    cost_range = df_candidates["avg_cost"].max() - df_candidates["avg_cost"].min()
    lat_range = df_candidates["avg_latency_ms"].max() - df_candidates["avg_latency_ms"].min()

    df_candidates["norm_accuracy"] = (df_candidates["avg_accuracy"] - df_candidates["avg_accuracy"].min()) / max(acc_range, 1e-9)
    df_candidates["norm_cost"] = 1 - (df_candidates["avg_cost"] - df_candidates["avg_cost"].min()) / max(cost_range, 1e-9)  # Lower is better
    df_candidates["norm_latency"] = 1 - (df_candidates["avg_latency_ms"] - df_candidates["avg_latency_ms"].min()) / max(lat_range, 1e-9)  # Lower is better

    df_candidates["composite_score"] = (
        weights["accuracy"] * df_candidates["norm_accuracy"] +
        weights["cost"] * df_candidates["norm_cost"] +
        weights["latency"] * df_candidates["norm_latency"]
    )

    df_candidates = df_candidates.sort_values("composite_score", ascending=False)

    # 4. Build recommendations
    recommendations = []
    for _, row in df_candidates.head(top_n).iterrows():
        rec = {
            "model_id": row["model_id"],
            "provider": row["provider"],
            "avg_accuracy": row["avg_accuracy"],
            "avg_cost": round(row["avg_cost"], 6),
            "avg_latency_ms": round(row["avg_latency_ms"], 1),
            "composite_score": round(row["composite_score"], 4),
            "sample_count": int(row["sample_count"]),
        }
        recommendations.append(rec)

    # 5. Compute delta if current_model is specified
    delta = None
    if current_model and current_model in profiles:
        current_stats = profiles[current_model].get("by_complexity", {}).get(complexity, None)
        if current_stats and recommendations:
            best = recommendations[0]
            delta = {
                "from_model": current_model,
                "to_model": best["model_id"],
                "accuracy_delta": round(best["avg_accuracy"] - current_stats["avg_accuracy"], 2),
                "cost_delta": round(best["avg_cost"] - current_stats["avg_cost"], 6),
                "cost_delta_pct": round(
                    ((best["avg_cost"] - current_stats["avg_cost"]) / max(current_stats["avg_cost"], 1e-9)) * 100, 1
                ),
                "latency_delta_ms": round(best["avg_latency_ms"] - current_stats["avg_latency_ms"], 1),
                "latency_delta_pct": round(
                    ((best["avg_latency_ms"] - current_stats["avg_latency_ms"]) / max(current_stats["avg_latency_ms"], 1e-9)) * 100, 1
                ),
            }

    # 6. If no current model specified, compute delta vs. worst model
    if delta is None and len(df_candidates) > 1 and recommendations:
        worst = df_candidates.iloc[-1]
        best = recommendations[0]
        delta = {
            "from_model": worst["model_id"],
            "to_model": best["model_id"],
            "accuracy_delta": round(best["avg_accuracy"] - worst["avg_accuracy"], 2),
            "cost_delta": round(best["avg_cost"] - worst["avg_cost"], 6),
            "cost_delta_pct": round(
                ((best["avg_cost"] - worst["avg_cost"]) / max(worst["avg_cost"], 1e-9)) * 100, 1
            ),
            "latency_delta_ms": round(best["avg_latency_ms"] - worst["avg_latency_ms"], 1),
            "latency_delta_pct": round(
                ((best["avg_latency_ms"] - worst["avg_latency_ms"]) / max(worst["avg_latency_ms"], 1e-9)) * 100, 1
            ),
        }

    return {
        "classification": classification,
        "recommendations": recommendations,
        "delta": delta,
        "all_models_count": len(df_candidates),
    }


def format_recommendation(result: dict) -> str:
    """Format the recommendation result into a human-readable string"""
    lines = []
    cls = result["classification"]
    lines.append(f"📋 Prompt Complexity: {cls['complexity'].upper()} (confidence: {cls['confidence']:.1%})")
    lines.append(f"   Probabilities: {cls['probabilities']}")
    lines.append("")

    if "error" in result:
        lines.append(f"⚠️  {result['error']}")
        return "\n".join(lines)

    lines.append(f"🏆 Top {len(result['recommendations'])} Recommended Models ({result['all_models_count']} evaluated):")
    lines.append("-" * 75)

    for i, rec in enumerate(result["recommendations"], 1):
        lines.append(
            f"  {i}. {rec['provider']}/{rec['model_id']}"
        )
        lines.append(
            f"     Accuracy: {rec['avg_accuracy']:.1f}/100  |  "
            f"Cost: ${rec['avg_cost']:.6f}  |  "
            f"Latency: {rec['avg_latency_ms']:.0f}ms  |  "
            f"Score: {rec['composite_score']:.4f}  (n={rec['sample_count']})"
        )

    if result.get("delta"):
        d = result["delta"]
        lines.append("")
        lines.append("=" * 75)

        # Build the delta string
        acc_sign = "+" if d["accuracy_delta"] >= 0 else ""
        cost_sign = "+" if d["cost_delta_pct"] >= 0 else ""
        lat_sign = "+" if d["latency_delta_ms"] >= 0 else ""

        lines.append(
            f"💡 Recommendation: Switching from {d['from_model']} to {d['to_model']} gives you:"
        )
        lines.append(
            f"   {acc_sign}{d['accuracy_delta']:.1f} accuracy  •  "
            f"{cost_sign}{d['cost_delta_pct']:.0f}% cost  •  "
            f"{lat_sign}{d['latency_delta_ms']:.0f}ms latency"
        )
        lines.append("=" * 75)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Interface
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM Model Recommendation System")
    parser.add_argument("--train", action="store_true", help="Train classifier and build profiles")
    parser.add_argument("--prompt", type=str, help="Classify and recommend for a single prompt")
    parser.add_argument("--current-model", type=str, default=None, help="Current model for delta comparison")
    parser.add_argument("--top-n", type=int, default=3, help="Number of recommendations")
    args = parser.parse_args()

    if args.train:
        # Training pipeline
        pipeline = train_classifier()
        profiles = build_model_profiles()

        # Demo predictions
        print("\n" + "=" * 60)
        print("DEMO: Sample Predictions")
        print("=" * 60)
        test_prompts = [
            "What is the capital of France?",
            "Compare remote work and office work in terms of productivity",
            "Design a distributed consensus algorithm that handles Byzantine fault tolerance for a 50-node cluster",
            "Write a Python function that reverses a string",
            "Given a flawed machine learning pipeline where data leakage occurs, propose a corrected architecture",
            "Explain photosynthesis in simple terms",
        ]
        for p in test_prompts:
            result = recommend(p, pipeline=pipeline, profiles=profiles)
            print(f"\n{'─' * 75}")
            print(f"Prompt: \"{p[:80]}{'...' if len(p) > 80 else ''}\"")
            print(format_recommendation(result))

    elif args.prompt:
        result = recommend(args.prompt, current_model=args.current_model, top_n=args.top_n)
        print(format_recommendation(result))

    else:
        # Interactive mode
        print("🤖 LLM Model Recommendation System")
        print("=" * 40)
        print("Enter a prompt to get model recommendations.")
        print("Type 'quit' to exit.\n")

        pipeline = load_classifier()
        profiles = load_profiles()

        while True:
            prompt = input("\n📝 Enter prompt: ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                break
            if not prompt:
                continue

            current = input("🔧 Current model (optional, press Enter to skip): ").strip() or None

            result = recommend(prompt, current_model=current, pipeline=pipeline, profiles=profiles)
            print("\n" + format_recommendation(result))


if __name__ == "__main__":
    main()

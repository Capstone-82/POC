# Prompt Profiling Engine -- PCA Experiment Summary & Improvement Roadmap

## 1. Objective

Evaluate whether increasing the PCA dimensionality from **40** to **50**
improves the Prompt Profiling Engine.

Both final comparisons use **5-Fold Stratified Cross-Validation**,
making them directly comparable.

------------------------------------------------------------------------

# 2. PCA Comparison

  Metric                              PCA-40       PCA-50    Delta Winner
  ----------------------------- ------------ ------------ -------- --------
  Blended Tier Accuracy           **81.29%**       81.16%   -0.13% PCA-40
  Direct Tier Accuracy            **80.79%**       80.63%   -0.16% PCA-40
  Formula Tier Accuracy           **79.88%**       79.76%   -0.12% PCA-40
  D1 Accuracy                         68.19%   **68.36%**   +0.17% PCA-50
  D2 Accuracy                     **71.21%**       69.93%   -1.28% PCA-40
  D3 Accuracy                     **76.21%**       75.63%   -0.58% PCA-40
  D4 Accuracy                     **80.13%**       79.14%   -0.99% PCA-40
  D5 Accuracy                         73.89%   **74.02%**   +0.13% PCA-50
  T3 Recall                         **0.82**         0.81    -0.01 PCA-40
  Accuracy @ Confidence ≥0.80     **90.82%**   **90.82%**    0.00% Tie

------------------------------------------------------------------------

# 3. Observations

-   Increasing PCA dimensions from **40 → 50** produced **no meaningful
    improvement**.
-   Tier accuracy changed by less than **0.2%**.
-   Confidence calibration remained identical.
-   D1 and D5 improved slightly.
-   D2, D3 and D4 became slightly worse.
-   PCA-40 remains the better configuration overall because it achieves
    marginally higher routing accuracy while using fewer dimensions.

------------------------------------------------------------------------

# 4. Why Accuracy Plateaued

The bottleneck is **not PCA or XGBoost** anymore.

The current architecture is already strong:

-   MiniLM Embeddings
-   PCA
-   50 Handcrafted Features
-   Shared Backbone
-   Multi-head XGBoost
-   Two-stage D-score conditioning
-   Blended Tier Prediction

The limiting factor is now primarily:

1.  Dataset quality
2.  Label consistency
3.  Embedding quality
4.  Boundary examples

------------------------------------------------------------------------

# 5. Recommended Roadmap

## Priority 1 (Highest ROI)

### 1. Upgrade embeddings

Benchmark: - BAAI/bge-large-en-v1.5 - intfloat/e5-large-v2 -
jina-embeddings-v3 - Nomic Embed

Expected gain: **+2% to +4%**

------------------------------------------------------------------------

### 2. Error-driven augmentation

Generate new prompts specifically for confusion cases such as:

-   D1: 0.50 vs 0.75
-   D2: 0.50 vs 0.75
-   T2 vs T3 boundary

Expected gain: **+2% to +5%**

------------------------------------------------------------------------

### 3. Audit labels

Review all: - incorrect predictions - low-confidence samples - boundary
prompts

Fix inconsistent annotations.

Expected gain: **+2% to +6%**

------------------------------------------------------------------------

### 4. Retrieval-Augmented Classification

Pipeline:

Prompt → Embedding → FAISS Retrieval → Similar Prompt Features →
Classifier

Expected gain: **+2% to +5%**

------------------------------------------------------------------------

### 5. Ordinal learning for D1--D5

Treat D scores as ordered values instead of unrelated classes.

Expected gain: **+1% to +3%**

------------------------------------------------------------------------

# Secondary Improvements

-   Multi-embedding fusion
-   Stacking (XGBoost + CatBoost + LightGBM)
-   Optuna hyperparameter optimization
-   Better probability calibration
-   Improved handcrafted features
-   Remove or relax PCA using explained variance instead of fixed
    dimensions

------------------------------------------------------------------------

# Dataset Recommendation

Current dataset: - 2,421 prompts

Recommended: - 10k--15k prompts - More real prompts - Better T2/T3
balance - More D1/D2 boundary samples

------------------------------------------------------------------------

# Long-Term Direction

Use an LLM **only offline** to generate high-quality labels and
explanations.

Workflow:

Prompt → GPT → D-score reasoning → High-quality labels → Train classical
ML model

This keeps inference fast while improving supervision quality.

------------------------------------------------------------------------

# Conclusion

Current performance (\~81% blended tier accuracy) is close to the limit
of the present pipeline.

Future gains are unlikely to come from further PCA tuning.

The highest-impact work is:

1.  Better embeddings
2.  Cleaner labels
3.  Retrieval augmentation
4.  Error-driven dataset expansion
5.  Ordinal prediction for D1--D5

These changes provide the best opportunity to move toward **87--90% raw
accuracy** while preserving a non-LLM inference pipeline.

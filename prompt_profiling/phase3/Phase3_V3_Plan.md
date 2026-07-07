# Phase 3 Plan — Single-Stage Ordinal & Retrieval-Augmented Model (V3)

## Goals
Evaluate a new simplified pipeline that resolves the remaining accuracy bottlenecks without the complexity of two-stage models.

## Proposed Changes

### [Component Name] Model Architecture V3

#### [NEW] [phase3_a1_v3.ipynb](file:///c:/Users/Musharraf/Documents/POC/prompt_profiling/phase3/phase3_a1_v3.ipynb)

We will implement three enhancements in this new notebook:

1. **Frank & Hall Ordinal Classifiers for D1–D5**:
   Instead of multi-class XGBoost models that treat D-scores (0.0, 0.25, 0.50, 0.75, 1.0) as unrelated categories, we train 4 binary classifiers for each dimension representing progressive thresholds ($D_i \ge 0.25$, $D_i \ge 0.50$, $D_i \ge 0.75$, $D_i \ge 1.00$).

2. **Expected Value D-Scores**:
   Compute expected values from the ordinal probabilities:
   $$E[D_i] = \sum_{c} \text{score}_c \times P(D_i = \text{score}_c)$$
   This continuous value is plugged directly into the tier formula:
   $$\text{ComplexityScore} = \sum w_i E[D_i]$$
   This prevents sudden metric jumps near the 0.40 and 0.70 thresholds and resolves boundary discretization errors.

3. **Cosine Nearest-Neighbor Features**:
   Before classification, retrieve the 5 most semantically similar training samples from the BGE embedding space using `sklearn.neighbors.NearestNeighbors(metric='cosine')`. Add neighbor stats (`nn_tier_mean`, `nn_tier_std`, `nn_d1_mean`, `nn_d2_mean`, etc.) to the feature matrix.

---

## Verification Plan

### Manual Verification
- User will run the notebook [phase3_a1_v3.ipynb](file:///c:/Users/Musharraf/Documents/POC/prompt_profiling/phase3/phase3_a1_v3.ipynb) in Google Colab using the Phase 3 dataset copy and share the outputs.

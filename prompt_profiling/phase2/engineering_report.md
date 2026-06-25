# Phase 2 Prompt Profiling Engine - Engineering Report

**System:** Extended Non-LLM Prompt Profiling Classifier  
**Architecture:** Shared-Backbone Multi-Head XGBoost with D-Score Conditioning  
**Evaluation:** 5-Fold Stratified Cross-Validation on 2,421 Samples  
**Dataset:** Cleaned Phase 2 + Phase 1 Merge + Targeted Augmentation  
**Date:** June 2026  
**Status:** Final Phase 2 Candidate  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Inference Output Schema](#3-inference-output-schema)
4. [Dataset](#4-dataset)
5. [Scoring Framework](#5-scoring-framework)
6. [Architecture](#6-architecture)
7. [Implementation Details](#7-implementation-details)
8. [Evaluation Results](#8-evaluation-results)
9. [Confidence-Based Routing](#9-confidence-based-routing)
10. [Example Prediction](#10-example-prediction)
11. [Engineering Notes](#11-engineering-notes)
12. [Limitations](#12-limitations)
13. [Conclusion](#13-conclusion)

---

## 1. Executive Summary

Phase 2 extends the Phase 1 prompt complexity classifier into a richer prompt profiling engine. In addition to predicting the five rubric dimensions and routing tier, the Phase 2 model predicts intent, task type, reasoning-chain requirement, research signals, and a confidence score.

The final Phase 2 model is a non-LLM classifier. It uses a MiniLM sentence embedding backbone, PCA compression, hand-crafted prompt features, and multiple XGBoost classification heads. The model is intended for fast pre-routing inference before a prompt reaches an LLM.

**Final model results, 5-fold stratified cross-validation, 2,421 samples:**

| Output | Accuracy | Macro F1 |
|---|---:|---:|
| Tier, direct classifier | 80.79% +/- 1.17% | 81.42% +/- 0.88% |
| Tier, formula-derived | 80.13% +/- 1.37% | 80.86% +/- 1.66% |
| Tier, blended final candidate | **81.29% +/- 1.59%** | **82.00% +/- 1.43%** |
| Intent | 82.07% +/- 1.60% | 78.97% +/- 1.73% |
| Task type | 73.52% +/- 2.02% | 69.27% +/- 6.10% |
| Reasoning-chain detected | 89.92% +/- 0.16% | 85.40% +/- 0.64% |
| D1 - Semantic Complexity | 68.19% +/- 1.91% | 70.97% +/- 1.68% |
| D2 - Domain Specificity | 71.21% +/- 1.35% | 65.55% +/- 1.64% |
| D3 - Output Formality | 76.21% +/- 1.79% | 72.85% +/- 2.33% |
| D4 - Research Dependency | 80.13% +/- 1.46% | 70.71% +/- 1.88% |
| D5 - Context Requirement | 73.89% +/- 1.62% | 71.88% +/- 0.92% |

The raw final tier accuracy is approximately **81%**. The model also provides useful confidence gating:

| Tier Confidence Threshold | Accuracy | Coverage |
|---|---:|---:|
| >= 0.60 | 82.51% | 95.42% |
| >= 0.70 | 86.03% | 81.58% |
| >= 0.80 | **90.82%** | 62.12% |
| >= 0.90 | **95.97%** | 37.92% |

The recommended deployment stance is:

- use the blended tier as the final tier candidate;
- treat raw tier accuracy as approximately 81%;
- use confidence thresholds for routing guarantees;
- accept high-confidence predictions directly;
- conservatively handle low-confidence boundary cases.

---

## 2. Problem Statement

The engineering requirement for Phase 2 is broader than Phase 1. Phase 1 focused on predicting:

- `d1` through `d5`
- `complexity_score`
- `tier`

Phase 2 requires a richer inference JSON that also includes:

- `intent`
- `task_type`
- `reasoning_chain_detected`
- `research_signals`
- `confidence`

This turns the classifier from a pure complexity-tier model into a prompt profiling engine. The system must still be:

1. **Fast** - suitable for routing-time inference
2. **Deterministic** - no LLM calls during inference
3. **Explainable** - returns dimension scores and routing signals
4. **Operationally simple** - can be serialized and deployed as standard ML artifacts
5. **Confidence-aware** - exposes when the tier prediction is reliable

The most important business output remains `tier`, because it determines which model pool should handle the prompt. The new outputs improve observability, routing policy, and downstream prompt handling.

---

## 3. Inference Output Schema

The final model returns the following output shape:

```json
{
  "d1": 1.0,
  "d1_label": "Semantic Complexity",
  "d2": 1.0,
  "d2_label": "Domain Specificity",
  "d3": 1.0,
  "d3_label": "Output Formality",
  "d4": 0.75,
  "d4_label": "Research Dependency",
  "d5": 0.75,
  "d5_label": "Context Requirement",
  "complexity_score": 0.9375,
  "tier": "T3",
  "direct_tier": "T3",
  "formula_tier": "T3",
  "intent": "STRATEGIC",
  "task_type": "reasoning",
  "reasoning_chain_detected": true,
  "research_signals": [
    "regulatory_compliance",
    "security",
    "cloud_infrastructure",
    "ai_governance",
    "vendor_analysis"
  ],
  "confidence": 0.9205,
  "tier_confidence": 0.9246
}
```

### 3.1 Field Definitions

| Field | Type | Description |
|---|---|---|
| `d1` | float | Semantic Complexity score |
| `d2` | float | Domain Specificity score |
| `d3` | float | Output Formality score |
| `d4` | float | Research Dependency score |
| `d5` | float | Context Requirement score |
| `complexity_score` | float | Weighted score computed from D1-D5 |
| `tier` | string | Final tier candidate, currently blended tier |
| `direct_tier` | string | Direct XGBoost tier-head prediction |
| `formula_tier` | string | Tier derived from predicted D1-D5 score formula |
| `intent` | string | Prompt intent classification |
| `task_type` | string | Task category classification |
| `reasoning_chain_detected` | boolean | Whether the prompt requires multi-step reasoning |
| `research_signals` | string array | Rule-based research/domain signals |
| `confidence` | float | Combined confidence across key output heads |
| `tier_confidence` | float | Confidence of final tier routing candidate |

### 3.2 Valid Values

Dimension scores:

```text
0.0, 0.25, 0.5, 0.75, 1.0
```

Tier:

```text
T1, T2, T3
```

Intent:

```text
FACTUAL
ANALYTICAL
SYNTHETIC
STRATEGIC
```

Task type:

```text
classification
coding
formatting
generation
reasoning
sparql_generation
summarisation
```

---

## 4. Dataset

### 4.1 Dataset Source

The engineering team provided a raw Phase 2 dataset containing 1,450 rows. The dataset included the new Phase 2 fields, but it was not directly training-ready. To build a usable final training dataset, the Phase 2 data was:

1. cleaned;
2. merged with the Phase 1 final dataset;
3. lightly augmented to address important missing score levels and rare task types.

The final Phase 2 training file is:

```text
prompt_profiling/phase2/prompt_classifier_phase2_v5_dataset.csv
```

The final modeling notebook is:

```text
prompt_profiling/phase2/phase2_a1_v5.ipynb
```

### 4.2 Issues in the Raw Phase 2 Dataset

The raw Phase 2 dataset had useful prompt diversity, but several issues had to be corrected before training:

| Issue | Impact |
|---|---|
| Missing D-score levels | Some dimension classes had no examples, so the model could not learn them |
| Severe class imbalance | D3, D4, and some task types were dominated by one class |
| T3 under-representation | Raw Phase 2 had very few T3 examples |
| Invalid `intent` values | Values such as `GENERATION`, `generation`, `CLASSIFICATION`, and mixed labels had to be remapped |
| Non-standard `task_type` values | Values such as `translation`, `explanation`, and pipe-separated labels had to be normalized |
| `reasoning_chain_detected` stored as string | Converted to boolean |
| `research_signals` stored as stringified lists | Parsed and normalized |
| Raw `complexity` did not match D-score-derived tier | D-scores were treated as ground truth and tier was re-derived |
| Duplicate prompts | Removed to prevent train/test leakage |
| Low-confidence labels | Retained but flagged |

### 4.3 Phase 1 Merge

The Phase 1 final dataset was merged because it contained strong coverage of:

- all five D-score levels for every dimension;
- T3 examples;
- enterprise-style prompts;
- boundary-zone examples;
- Phase 1 phrasing styles and domains.

Phase 1 rows did not originally contain Phase 2 fields such as `intent`, `task_type`, and `research_signals`, so those fields were derived using deterministic rules.

Examples:

| Field | Phase 1 Derivation |
|---|---|
| `intent` | Derived from D1 |
| `task_type` | Derived from prompt-text heuristics |
| `reasoning_chain_detected` | `True` when D1 >= 0.50 |
| `research_signals` | Derived from D4 and keyword/domain signals |
| `confidence` | Null, because Phase 1 did not include confidence labels |
| `source` | `phase1` |

The original source CSVs were not modified.

### 4.4 Final Dataset Shape

| Property | Value |
|---|---:|
| Total rows | 2,421 |
| Original rows retained | 2,273 |
| Added final augmentation rows | 48 |
| Total augmentation share from final v5 rows | 1.98% |
| Duplicate prompts | 0 |
| Tier formula mismatches | 0 |

The final dataset contains rows from:

| Source | Rows |
|---|---:|
| Phase 2 cleaned data | 1,447 |
| Phase 1 merged data | 826 |
| Earlier targeted augmentation retained in v4 | 100 |
| Final D1/D2-focused augmentation | 48 |

### 4.5 Tier Distribution

| Tier | Count |
|---|---:|
| T1 | 970 |
| T2 | 1,072 |
| T3 | 379 |

The dataset remains T1/T2-heavy, which reflects the original Phase 2 distribution. T3 coverage was improved, but not artificially over-balanced.

### 4.6 D-Score Distributions

| Score | D1 | D2 | D3 | D4 | D5 |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 468 | 128 | 298 | 1,728 | 1,051 |
| 0.25 | 109 | 140 | 143 | 145 | 329 |
| 0.50 | 791 | 1,377 | 1,314 | 291 | 849 |
| 0.75 | 856 | 568 | 670 | 262 | 244 |
| 1.00 | 197 | 208 | 208 | 207 | 160 |

Notable observations:

- D1 and D2 are the most important dimensions for tier because they carry the highest weights: 0.35 and 0.20.
- Phase 2 raw data had poor coverage for some D1/D2 score levels.
- The final augmentation specifically added examples around D1=0.25, D1=1.0, D2=0.0, and D2=1.0.
- D4 remains zero-heavy, which is expected because most prompts do not require external research.

### 4.7 Task Type Distribution

| Task Type | Count |
|---|---:|
| reasoning | 1,314 |
| generation | 782 |
| summarisation | 120 |
| classification | 100 |
| coding | 87 |
| formatting | 10 |
| sparql_generation | 8 |

`formatting` and `sparql_generation` are still low-support classes. They are included in the final schema and model, but their per-class metrics should be interpreted cautiously.

### 4.8 Dataset Design Principle

The final dataset was not aggressively augmented to chase high headline accuracy. The purpose of augmentation was targeted coverage repair:

- missing D1/D2 score levels;
- rare task types;
- T2/T3 boundary examples;
- Phase 2-style T3 examples.

This preserves a realistic evaluation profile. The final model is therefore not claiming inflated 90%+ raw accuracy from synthetic over-representation.

---

## 5. Scoring Framework

The Phase 2 model keeps the Phase 1 complexity formula.

```text
ComplexityScore =
  (D1 * 0.35) +
  (D2 * 0.20) +
  (D3 * 0.20) +
  (D4 * 0.15) +
  (D5 * 0.10)
```

Tier thresholds:

| Tier | Score Range | Interpretation |
|---|---|---|
| T1 | 0.00 - 0.39 | Simple |
| T2 | 0.40 - 0.69 | Medium |
| T3 | 0.70 - 1.00 | Complex |

### 5.1 Dimension Definitions

| Dimension | Label | Weight | Description |
|---|---|---:|---|
| D1 | Semantic Complexity | 0.35 | Reasoning depth, synthesis, strategic complexity |
| D2 | Domain Specificity | 0.20 | Specialized domain knowledge, tools, vendors, frameworks |
| D3 | Output Formality | 0.20 | Expected structure, length, and deliverable formality |
| D4 | Research Dependency | 0.15 | Need for external/live/current information |
| D5 | Context Requirement | 0.10 | Required input context size and artifact dependency |

### 5.2 Why D1 and D2 Matter Most

D1 has the largest tier weight, and D2 has the second-highest weight tied with D3. A one-step error in D1 changes the complexity score by:

```text
0.25 * 0.35 = 0.0875
```

A one-step error in D2 changes the score by:

```text
0.25 * 0.20 = 0.0500
```

Near the T2/T3 boundary at 0.70, these errors are enough to flip the tier. This is why the final dataset and model pay special attention to D1/D2.

---

## 6. Architecture

### 6.1 High-Level Architecture

```text
Prompt
  |
  |--> MiniLM Sentence Embedding, 384 dimensions
  |
  |--> PCA compression, 40 dimensions
  |
  |--> Hand-crafted prompt features, 50 dimensions
  |
  v
Shared Feature Matrix, 90 dimensions
  |
  |--> Stage 1 heads:
  |      - tier
  |      - intent
  |      - task_type
  |      - reasoning_chain_detected
  |
  |--> Stage 2 D-score heads:
  |      - D1
  |      - D2
  |      - D3
  |      - D4
  |      - D5
  |      input = shared features + predicted tier
  |
  |--> Formula-derived tier from predicted D1-D5
  |
  |--> Blended tier candidate
  |
  |--> Rule-based research_signals
```

### 6.2 Backbone

The backbone combines semantic and explicit prompt features:

| Component | Output |
|---|---:|
| MiniLM embedding | 384 dimensions |
| PCA-reduced embedding | 40 dimensions |
| Hand-crafted features | 50 dimensions |
| Final shared feature vector | 90 dimensions |

The embedding model used in the notebook is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### 6.3 Hand-Crafted Features

The final model uses 50 hand-crafted features. These are grouped as:

| Feature Group | Purpose |
|---|---|
| Text statistics | Length, sentence count, uniqueness, line count |
| Context requirement features | Attachment, artifact, large-context signals |
| Output formality features | Formal deliverable, report package, structured sections |
| Semantic complexity features | Strategic wording, action verbs, multi-stage requests |
| Domain specificity features | Compliance, cloud providers, systems, frameworks |
| Research dependency features | Time references, external data, vendor tools, market terms |
| Boundary/risk features | Comparison, stakeholders, risk language |
| Phase 2 task/intent features | Role prompts, step requests, code blocks, output formats |
| D1-specific features | Strategic signal count, simple factual signal, solution design signal |
| D2-specific features | Domain term count, acronym count, vendor/framework signal |
| Phase 1 phrasing style features | explicit, implicit, vague one-hot features |

The final D1/D2-specific features were added because these dimensions have the largest effect on tier routing.

### 6.4 Model Heads

The model trains separate XGBoost classifier heads:

| Head | Classes |
|---|---:|
| Tier | 3 |
| Intent | 4 |
| Task type | 7 |
| Reasoning-chain detected | 2 |
| D1 | 5 |
| D2 | 5 |
| D3 | 5 |
| D4 | 5 |
| D5 | 5 |

### 6.5 Two-Stage D-Score Conditioning

D-score heads receive the predicted tier as an additional feature.

```text
Stage 1:
  Train tier, intent, task_type, reasoning_chain heads

Stage 2:
  Append predicted tier to shared feature matrix
  Train D1-D5 heads on augmented feature matrix
```

To reduce train/evaluation mismatch, out-of-fold tier predictions are used for D-score training during cross-validation.

### 6.6 Blended Tier

The model computes three tier-related outputs:

1. `direct_tier` - direct XGBoost tier classifier
2. `formula_tier` - derived from predicted D1-D5 scores
3. `tier` - blended final candidate

The blended candidate combines the direct tier probability with the formula-derived tier signal:

```text
blended_proba = 0.72 * direct_tier_proba + 0.28 * formula_tier_onehot
```

The final `tier` field uses this blended prediction.

---

## 7. Implementation Details

### 7.1 XGBoost Configuration

The final notebook uses:

```python
n_estimators = 350
max_depth = 3
learning_rate = 0.045
subsample = 0.88
colsample_bytree = 0.88
reg_lambda = 2.5
min_child_weight = 5
tree_method = "hist"
```

Rationale:

- shallow trees reduce overfitting;
- `min_child_weight=5` discourages memorizing rare classes;
- row and column subsampling reduce variance;
- regularization is intentionally conservative;
- the final dataset uses only light augmentation, so model capacity is kept controlled.

### 7.2 Sample Weighting

Balanced sample weights are capped per target to avoid over-correcting rare classes.

| Target | Weight Cap |
|---|---:|
| tier | 2.2 |
| intent | 3.0 |
| task_type | 2.5 |
| reasoning_chain_detected | 2.0 |
| D1 | 4.0 |
| D2 | 4.0 |
| D3 | 3.0 |
| D4 | 3.0 |
| D5 | 3.0 |

D1 and D2 receive slightly stronger caps because they have the largest influence on tier.

### 7.3 Research Signals

`research_signals` are extracted with a rule engine rather than treated as a primary learned multi-label model. This is intentional because the raw Phase 2 signal labels were not consistently aligned with D4 at first.

Rule behavior:

- if predicted D4 is 0, return `[]`;
- if predicted D4 is greater than 0, scan for research/domain keywords;
- if no specific signal matches, return `["external_research"]`.

Signal categories include:

- `market_research`
- `competitive_analysis`
- `regulatory_compliance`
- `security`
- `cloud_infrastructure`
- `finops`
- `devops`
- `data_engineering`
- `ai_governance`
- `system_integration`
- `supply_chain`
- `hr_tech`
- `vendor_analysis`

### 7.4 Inference Pipeline

```text
1. Receive prompt
2. Generate MiniLM embedding
3. Apply PCA transform
4. Extract hand-crafted features
5. Scale combined feature vector
6. Predict direct tier
7. Append predicted tier to feature vector
8. Predict D1-D5
9. Compute formula-derived tier
10. Blend direct tier and formula tier
11. Predict intent, task_type, reasoning_chain
12. Extract research signals
13. Return final JSON
```

---

## 8. Evaluation Results

### 8.1 Methodology

Evaluation uses 5-fold stratified cross-validation. Stratification is by tier.

Metrics are reported as mean +/- standard deviation across folds.

The report includes:

- direct tier;
- formula-derived tier;
- blended tier;
- D1-D5;
- intent;
- task type;
- reasoning-chain detection;
- confidence threshold behavior.

### 8.2 Final Output Metrics

| Output | Accuracy | Macro F1 |
|---|---:|---:|
| Tier, direct | 80.79% +/- 1.17% | 81.42% +/- 0.88% |
| Tier, formula-derived | 80.13% +/- 1.37% | 80.86% +/- 1.66% |
| Tier, blended final candidate | **81.29% +/- 1.59%** | **82.00% +/- 1.43%** |
| Intent | 82.07% +/- 1.60% | 78.97% +/- 1.73% |
| Task type | 73.52% +/- 2.02% | 69.27% +/- 6.10% |
| Reasoning-chain detected | 89.92% +/- 0.16% | 85.40% +/- 0.64% |
| D1 | 68.19% +/- 1.91% | 70.97% +/- 1.68% |
| D2 | 71.21% +/- 1.35% | 65.55% +/- 1.64% |
| D3 | 76.21% +/- 1.79% | 72.85% +/- 2.33% |
| D4 | 80.13% +/- 1.46% | 70.71% +/- 1.88% |
| D5 | 73.89% +/- 1.62% | 71.88% +/- 0.92% |

### 8.3 Tier Accuracy by Evaluation Slice

The slice metrics are important because the final dataset contains multiple sources. This prevents synthetic augmentation from hiding original-data performance.

| Slice | Direct Tier Accuracy | Blended Tier Accuracy |
|---|---:|---:|
| All rows | 80.79% +/- 1.17% | **81.29% +/- 1.59%** |
| Original rows only | 79.98% +/- 1.32% | **80.47% +/- 1.78%** |
| Phase 1 rows | 88.00% +/- 1.79% | **88.66% +/- 2.00%** |
| Phase 2 rows | 75.43% +/- 1.56% | **75.78% +/- 1.68%** |
| v4 augmentation rows | 95.87% +/- 3.80% | 95.87% +/- 3.80% |
| v5 augmentation rows | 84.09% +/- 9.78% | **85.90% +/- 11.57%** |
| T2/T3 boundary rows | **84.68% +/- 3.46%** | 81.40% +/- 4.45% |

Interpretation:

- Phase 1-style enterprise rows are close to the target band.
- Phase 2 original rows remain the hardest distribution.
- The blended tier improves all-row and original-only accuracy slightly.
- Direct tier performs better on the T2/T3 boundary slice.
- The final deployment can retain both `tier` and `direct_tier` for observability.

### 8.4 Tier Confusion Matrix

Blended tier confusion matrix:

```text
Predicted ->    T1    T2    T3
Actual
T1            794   171     5
T2            168   864    40
T3              3    66   310
```

Tier classification report:

| Tier | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| T1 | 0.82 | 0.82 | 0.82 | 970 |
| T2 | 0.78 | 0.81 | 0.80 | 1,072 |
| T3 | 0.87 | 0.82 | 0.84 | 379 |

The most important safety observation is that T3 precision is strong at 0.87, and T3 recall is 0.82. T3-to-T1 errors are nearly eliminated: only 3 cases in the aggregate cross-validation confusion matrix.

### 8.5 D1 Performance

| D1 Score | Precision | Recall | F1 | Support |
|---:|---:|---:|---:|---:|
| 0.00 | 0.62 | 0.69 | 0.65 | 468 |
| 0.25 | 0.72 | 0.81 | 0.76 | 109 |
| 0.50 | 0.66 | 0.56 | 0.61 | 791 |
| 0.75 | 0.72 | 0.72 | 0.72 | 856 |
| 1.00 | 0.73 | 0.93 | 0.82 | 197 |

Overall:

```text
D1 accuracy: 68.19%
D1 macro F1: 70.97%
```

Interpretation:

- D1=0.25 and D1=1.0 are now recognized well.
- The main confusion remains between D1=0.50 and D1=0.75.
- This is expected because many Phase 2 analytical and synthetic prompts overlap in style.

### 8.6 D2 Performance

| D2 Score | Precision | Recall | F1 | Support |
|---:|---:|---:|---:|---:|
| 0.00 | 0.42 | 0.60 | 0.50 | 128 |
| 0.25 | 0.47 | 0.74 | 0.58 | 140 |
| 0.50 | 0.86 | 0.71 | 0.78 | 1,377 |
| 0.75 | 0.61 | 0.67 | 0.64 | 568 |
| 1.00 | 0.72 | 0.87 | 0.79 | 208 |

Overall:

```text
D2 accuracy: 71.21%
D2 macro F1: 65.55%
```

Interpretation:

- D2=1.0 recall is strong at 0.87.
- D2=0.0 and D2=0.25 are better detected than before, but precision is still modest.
- D2=0.50 dominates the dataset, and some borderline domain prompts are pulled toward adjacent classes.

### 8.7 Task Type Performance

| Task Type | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| classification | 0.48 | 0.53 | 0.50 | 100 |
| coding | 0.58 | 0.52 | 0.55 | 87 |
| formatting | 1.00 | 0.80 | 0.89 | 10 |
| generation | 0.68 | 0.61 | 0.64 | 782 |
| reasoning | 0.78 | 0.85 | 0.81 | 1,314 |
| sparql_generation | 1.00 | 0.75 | 0.86 | 8 |
| summarisation | 0.83 | 0.62 | 0.71 | 120 |

Overall:

```text
Task type accuracy: 73.52%
Task type macro F1: 69.27%
```

Interpretation:

- `reasoning` is strong because it is the dominant and most semantically broad class.
- `classification` and `coding` remain moderate.
- `formatting` and `sparql_generation` appear strong but have very low support.
- Task type should be treated as useful metadata, not as a routing-critical signal at this stage.

---

## 9. Confidence-Based Routing

The final model's most useful operational behavior is confidence gating.

### 9.1 Confidence Results

For blended tier:

| Confidence Threshold | Accuracy | Coverage |
|---|---:|---:|
| >= 0.60 | 82.51% | 95.42% |
| >= 0.70 | 86.03% | 81.58% |
| >= 0.80 | **90.82%** | 62.12% |
| >= 0.90 | **95.97%** | 37.92% |

This means the raw tier accuracy is approximately 81%, but when the model is confident, accuracy is substantially higher.

### 9.2 Recommended Routing Policy

The recommended production policy is:

```text
if tier_confidence >= 0.80:
    accept predicted tier
elif tier_confidence >= 0.70:
    accept predicted tier, but log as medium confidence
else:
    apply conservative fallback
```

For low-confidence prompts:

- if direct tier and formula tier agree, accept but log;
- if direct tier and formula tier disagree near the T2/T3 boundary, route conservatively to T3;
- if the prompt is out-of-distribution or business-critical, send for review or premium routing.

This policy allows the system to make high-confidence decisions where reliable while reducing under-routing risk for ambiguous prompts.

---

## 10. Example Prediction

Example prompt:

```text
Design a multi-cloud GenAI governance architecture for a Fortune 500 company,
including compliance risks and vendor evaluation criteria.
```

Model output:

```json
{
  "d1": 1.0,
  "d1_label": "Semantic Complexity",
  "d2": 1.0,
  "d2_label": "Domain Specificity",
  "d3": 1.0,
  "d3_label": "Output Formality",
  "d4": 0.75,
  "d4_label": "Research Dependency",
  "d5": 0.75,
  "d5_label": "Context Requirement",
  "complexity_score": 0.9375,
  "tier": "T3",
  "direct_tier": "T3",
  "formula_tier": "T3",
  "intent": "STRATEGIC",
  "task_type": "reasoning",
  "reasoning_chain_detected": true,
  "research_signals": [
    "regulatory_compliance",
    "security",
    "cloud_infrastructure",
    "ai_governance",
    "vendor_analysis"
  ],
  "confidence": 0.9205,
  "tier_confidence": 0.9246
}
```

Interpretation:

- D1=1.0 because the prompt asks for strategic architecture design.
- D2=1.0 because it combines multi-cloud, GenAI governance, compliance, and vendor evaluation.
- D3=1.0 because the expected output is architecture-level and formal.
- D4=0.75 because vendor evaluation and compliance risks imply external/current information.
- D5=0.75 because the prompt likely requires significant context and structured analysis.
- Final tier is T3 with high confidence.

---

## 11. Engineering Notes

### 11.1 Files

Final Phase 2 dataset:

```text
prompt_profiling/phase2/prompt_classifier_phase2_v5_dataset.csv
```

Final Phase 2 notebook:

```text
prompt_profiling/phase2/phase2_a1_v5.ipynb
```

Dataset audit:

```text
prompt_profiling/phase2/previous_versions/v5/V5_DATASET_AUDIT.md
```

Dataset generation script:

```text
prompt_profiling/phase2/previous_versions/v5/create_v5_dataset.py
```

### 11.2 Artifacts Expected for Deployment

The notebook currently trains in Colab. For deployment, the following should be serialized:

| Artifact | Purpose |
|---|---|
| SentenceTransformer model reference | Text embedding |
| PCA transformer | 384-dim to 40-dim compression |
| StandardScaler | Feature scaling |
| XGBoost tier head | Direct tier prediction |
| XGBoost intent head | Intent prediction |
| XGBoost task_type head | Task type prediction |
| XGBoost reasoning head | Reasoning-chain detection |
| XGBoost D1-D5 heads | Dimension score prediction |
| Label encoders | Class-to-label mapping |
| Feature extraction code | Hand-crafted feature generation |
| Research signal rule engine | Research signal extraction |

### 11.3 Runtime Considerations

The runtime cost is dominated by MiniLM embedding. XGBoost inference is lightweight.

Expected inference path:

```text
embedding -> PCA -> hand features -> scaling -> XGBoost heads -> JSON output
```

No LLM call is required during classification.

### 11.4 Monitoring Recommendations

The following should be logged for production monitoring:

- prompt hash or request ID;
- predicted `tier`;
- `direct_tier`;
- `formula_tier`;
- `tier_confidence`;
- D1-D5 scores;
- `intent`;
- `task_type`;
- `source` not applicable in production, but similar drift bucket can be added;
- whether fallback policy was triggered.

Useful monitoring metrics:

- tier distribution over time;
- percentage of prompts below confidence 0.70;
- percentage of T3 predictions;
- direct tier vs formula tier disagreement rate;
- prompt length distribution;
- domain/research signal frequency.

---

## 12. Limitations

### 12.1 Raw Accuracy Does Not Reach 90%

The final raw blended tier accuracy is approximately 81%. The system reaches 90%+ only under confidence gating:

```text
tier_confidence >= 0.80 -> 90.82% accuracy at 62.12% coverage
```

This should be communicated clearly. The model is reliable when confident, but not a universal 90% classifier across all prompts.

### 12.2 Phase 2 Original Data Remains Hard

The Phase 2 original rows are the hardest slice:

```text
Phase 2 blended tier accuracy: 75.78%
```

The likely reasons are:

- broader domain diversity;
- noisier LLM-generated labels;
- sparse high-complexity Phase 2 examples;
- D1/D2 ambiguity in analytical vs synthetic prompts.

### 12.3 D1 and D2 Are Still Challenging

D1 and D2 are the highest-impact dimensions, but they remain imperfect:

```text
D1 accuracy: 68.19%
D2 accuracy: 71.21%
```

Minority D1/D2 classes improved, but mid-class confusion remains. This affects tier because D1 and D2 have high formula weights.

### 12.4 Task Type Is Useful but Not Routing-Critical

Task type accuracy is 73.52%. This is useful for metadata and analytics but should not be treated as a hard operational guarantee.

### 12.5 Synthetic Data Bias

The dataset includes LLM-generated and augmented rows. Although augmentation was kept small, synthetic phrasing can still introduce patterns that are easier for the model to learn than real production traffic.

### 12.6 English-First System

The model, keyword features, and training data are English-first. Multilingual or heavily code-switched prompts are not covered.

---

## 13. Conclusion

The final Phase 2 Prompt Profiling Engine extends the Phase 1 classifier into a richer, multi-output prompt profiling system. It predicts the original five complexity dimensions and tier, while also adding intent, task type, reasoning-chain detection, research signals, and confidence.

The final model achieves:

```text
Raw blended tier accuracy: 81.29%
Original-only blended tier accuracy: 80.47%
Phase 1 blended tier accuracy: 88.66%
Phase 2 blended tier accuracy: 75.78%
High-confidence tier accuracy at confidence >= 0.80: 90.82%
High-confidence coverage at confidence >= 0.80: 62.12%
```

The recommended engineering interpretation is:

- The model is not a universal 90% raw classifier.
- It is a usable routing classifier with strong high-confidence behavior.
- Confidence gating is essential for operational reliability.
- Low-confidence and boundary cases should be conservatively routed or reviewed.
- The system remains fully non-LLM at inference time and provides explainable dimension-level outputs.

For deployment, the safest policy is to use blended tier as the final candidate, accept high-confidence predictions directly, and apply conservative fallback behavior for low-confidence T2/T3 boundary prompts.

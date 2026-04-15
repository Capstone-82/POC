# LLM Model Router — Production Architecture (v2)

## System Overview
A recommendation engine that selects the best LLM using:
- Similarity search
- Rule filtering
- ML ranking

---

## Constraints
- No inference allowed
- <200 ms latency (P95)
- Multi-provider support
- Graceful fallback

---

## End-to-End Flow (6 Stages)

1. Prompt Analysis
2. Embedding
3. Similarity Search
4. Rule Engine
5. ML Router
6. Response Builder

### Parallel Execution
- Stage 1 & 2 → parallel
- Stage 4 & 5 → parallel

---

## Prompt Analyzer

Outputs:
- Complexity (low/mid/high)
- Quality score
- Intent tags
- Token estimate

---

## Embedding Service

- Model: 768 or 1536 dim
- Cache via SHA-256
- Optional PCA → 256 dim

---

## Similarity Search

- pgvector (HNSW)
- K = 50 default
- Threshold ≥ 0.75

---

## Signal Aggregation

Per model:
- avg_accuracy
- p50_cost
- p50_latency
- sample_count
- variance
- recency_weight

---

## Rule Engine

- Budget constraints
- Latency SLA
- Capability filtering
- Health checks
- Minimum sample filter

---

## ML Router (LightGBM)

### Features
- Prompt features
- Aggregated signals
- Model metadata
- Interaction features

### Output
- Ranked models
- Confidence score

---

## Decision Logic

Priority:
1. Rule Engine
2. ML Router (if confidence ≥ 0.7)
3. KNN fallback

---

## Scoring Formula

```
score = w_acc * accuracy
      + w_cost * (1 - cost)
      + w_lat * (1 - latency)
      + w_conf * confidence
```

Default:
- acc = 0.55
- cost = 0.25
- latency = 0.15
- conf = 0.05

---

## Accuracy Prediction

### Strategy 1 — Similarity Weighted
```
Σ(similarity × accuracy) / Σ(similarity)
```

### Strategy 2 — Regression
- Gradient boosting

### Strategy 3 — Confidence Intervals
- Low data → ±15
- High data → ±4

---

## Edge Cases

### Cold Start
- Expand search
- Use global priors

### New Model
- Seed benchmark
- Exploration routing (5%)

### Conflicting Scores
- Remove outliers
- Flag high conflict

### Capability Mismatch
- Override use_case

---

## Data & Storage

### Additions
- prompt_hash
- eval_conflict_flag
- model_version
- feedback_label

### Indexing
- HNSW (preferred)
- Partial index per use_case

---

## Learning Loop

### Flow
1. User interaction
2. Feedback collection
3. Labeling
4. Retraining
5. A/B testing

### Signal Weights
- Accept = +1.0
- Reject = -0.8

---

## Drift Detection

- Data drift → MMD
- Performance drift → acceptance rate

---

## Trade-offs

| KNN | ML Router |
|-----|----------|
| Works with low data | Needs scale |
| Interpretable | Learns interactions |

---

## Optimization

### Weight Presets
- Quality
- Cost
- Latency

---

## Advanced Ideas

- Cascade routing
- Bandit learning
- RL-based routing
- Prompt compression
- Conformal prediction

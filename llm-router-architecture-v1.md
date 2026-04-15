# LLM Model Router — Production Architecture (v1)

## Overview
The LLM Router is a **meta-intelligence layer** that selects the best LLM for a given prompt **without performing inference**.

### Key Properties
- Zero-inference system
- Multi-signal routing (KNN + Rules + ML)
- Feedback-driven learning
- Provider-agnostic
- Explainable decisions

---

## Output Contract
```json
{
  "recommended_model": "model_id",
  "provider": "provider_name",
  "expected_accuracy": 0.87,
  "estimated_cost": 0.0034,
  "estimated_latency": 1240,
  "confidence": 0.91,
  "rationale": {},
  "alternatives": []
}
```

---

## End-to-End Flow (8 Stages)

1. API Ingestion
2. Prompt Analyzer
3. Embedding + Cache
4. KNN Search
5. Rule Engine
6. ML Ranker
7. Score Fusion
8. Response Builder

---

## Core Components

| Component | Responsibility |
|----------|--------------|
| API Gateway | Validation, routing |
| Prompt Analyzer | Complexity, use-case, tokens |
| Embedding Service | Vector generation + caching |
| KNN Search | Similarity retrieval |
| Rule Engine | Hard constraints |
| ML Ranker | Learned ranking |
| Score Fusion | Final scoring |
| Response Builder | Output formatting |

---

## Decision Logic

### Layered System
1. **KNN (Baseline)**
2. **Rule Engine (Hard constraints)**
3. **ML Ranker (Refinement)**

### Scoring Formula
```
score = α·accuracy + β·(1 - cost) + γ·(1 - latency) + δ·ranker_score
```

Default weights:
- α = 0.50 (accuracy)
- β = 0.25 (cost)
- γ = 0.15 (latency)
- δ = 0.10 (ML)

---

## Accuracy Prediction

### Tier 1 — KNN Aggregation
Weighted similarity-based accuracy.

### Tier 2 — Priors
(model, use_case, complexity) lookup.

### Tier 3 — Regression Model
Gradient boosting on features.

---

## Edge Cases

### Cold Start
- Use priors
- Synthetic augmentation

### New Model
- Seed dataset (≥ 50 samples)
- Proxy model fallback

### OOD Prompt
- Expand K
- Use generalist model

---

## Data Schema (Simplified)

### benchmark_prompts
- model_id
- prompt
- accuracy
- cost
- latency

### prompt_embeddings
- embedding vector (pgvector)

### models
- cost
- latency
- capabilities

### routing_log
- decisions
- predictions

### feedback
- actual performance

---

## Learning Loop

### Feedback Sources
- User ratings
- Automated eval
- Implicit signals

### Update Cycles
- Real-time → data ingestion
- Hourly → recalibration
- Weekly → retraining

---

## Deployment

- Kubernetes microservices
- Kafka (async pipeline)
- Redis (cache)
- PostgreSQL + pgvector

---

## Trade-offs

| KNN | ML Ranker |
|-----|----------|
| No training needed | Needs data |
| Interpretable | Better accuracy |
| Fast | More complex |

---

## Extensions

- Mixture of Routers
- Bandit-based routing
- Cascade routing
- Uncertainty estimation

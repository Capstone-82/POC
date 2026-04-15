# Production LLM Router Architecture
### Tailored to the Existing POC Codebase

---

## PHASE 1 — CODEBASE ANALYSIS

### 1.1 System Overview

The current system is a **benchmark-driven model recommendation engine**. It does not use an LLM to pick a model. It uses historical benchmark data, a lightweight complexity classifier, and a tiered slice-matching strategy to recommend the best-value LLM for a given prompt.

Two distinct pipelines exist:

| Pipeline | Purpose |
|---|---|
| **Training Pipeline** | Accept prompts → call all provider models in parallel → Gemini evaluates responses → save rows to Supabase |
| **Inference Pipeline** | Accept prompt + use_case + current_model → infer complexity + clarity → slice benchmark data → rank → output recommendation |

These pipelines are **architecturally independent**. Training produces data. Inference consumes it. There is no live feedback from inference back into training.

---

### 1.2 Data Flow

#### Training Path (data generation)
```
Frontend (single prompt / CSV upload)
  → POST /api/training/run or /upload
  → training.py: process_prompts()
      ├── save_prompt_log()   → Supabase:prompt_logs
      ├── call_all_models()   → AWS Bedrock (14 models, parallel)
      ├── call_all_vertex_models() → Google Vertex (6 models, parallel)
      ├── evaluate_all_responses() → Gemini 2.5 Flash (round-robin pool, 23 endpoints)
      │     ↳ returns accuracy_score per model_id
      └── save_row()          → Supabase:benchmark_results
                                 { model_id, provider, use_case, prompt,
                                   prompt_complexity, clarity, response,
                                   accuracy_score, cost, tokens, latency_ms }
```

**Critical gap:** `avg_accuracy_score` is NOT written by the training pipeline. It is written later by the external script `generate_avg_accuracy_scores.py`. New rows land in Supabase with only `accuracy_score` populated.

#### Inference / Recommendation Path
```
Frontend (prompt + use_case + current_model)
  → POST /api/inference/recommend
  → recommender.py: get_recommendation()
      ├── load_complexity_classifier()      → classifier.pkl (TF-IDF + LogReg)
      ├── infer_complexity()                → low / mid / high
      ├── infer_clarity()                   → CLEAR / PARTIAL / UNCLEAR
      ├── load_benchmark_rows_with_fallback()
      │     ├── get_benchmark_data()        → Supabase (use-case filtered)
      │     └── fallback: benchmark_results.csv
      ├── clean_benchmark_rows()            ← reads accuracy_score (LEGACY)
      ├── filter_tiers (exact → complexity → use_case)
      ├── summarize_models()                → avg_accuracy, median_cost, median_latency
      ├── pick_best_value_model()           → 2-stage: quality shortlist → value score
      ├── should_switch()                   → switching policy
      └── return recommendation payload
```

**Critical gap:** `clean_benchmark_rows()` reads `accuracy_score` (the legacy, single-evaluator field), not `avg_accuracy_score`. The pipeline is not yet consuming the new multi-evaluator signal.

#### Offline Evaluation Pipeline (`generate_avg_accuracy_scores.py`)
```
CLI invocation (manual / scheduled)
  → fetch_rows() from Supabase (rows where avg_accuracy_score IS NULL)
  → for each row (parallel, ThreadPoolExecutor):
      ├── resolve_evaluators_for_use_case()
      │     text-generation: [llama4-maverick, mistral-large, nova-premier]
      │     code-generation: [llama4-maverick, mistral-large]
      │     reasoning:       [llama4-maverick, nova-premier]
      ├── evaluate_with_model() × N evaluators (sequential per row)
      │     → calls Bedrock or Vertex
      │     → retries up to 4× with exponential backoff
      │     → returns int score 0-100
      ├── store per-evaluator columns:
      │     eval_llama4_maverick_score
      │     eval_mistral_large_score
      │     eval_nova_premier_score
      │     eval_deepseek_r1_score
      ├── compute avg_accuracy_score:
      │     - include accuracy_score only if != 50 (placeholder guard)
      │     - simple mean of all available scores
      └── update_row() in Supabase
```

---

### 1.3 What Each File Does

| File | Role |
|---|---|
| `routers/training.py` | SSE-based job orchestrator for data collection |
| `routers/inference.py` | Thin API shim → calls recommender |
| `services/recommender.py` | **The router** — slice + rank + switch logic |
| `services/evaluator.py` | Gemini-based batch evaluator (training-time only) |
| `services/bedrock.py` | Parallel Bedrock caller (14 models) |
| `services/vertex.py` | Parallel Vertex caller (6 Gemini models) |
| `services/gemini_clients.py` | Round-robin pool across 23 Vertex endpoints |
| `services/model_registry.py` | Static model → use_case mapping |
| `services/supabase_client.py` | Supabase read/write with pagination |
| `model_training/generate_avg_accuracy_scores.py` | Offline multi-evaluator pipeline |
| `model_training/recommend_v2.py` | Standalone prototype (uses `avg_accuracy_score` already) |
| `model_training/Model_Arch.md` | Internal design doc for current recommender |

---

### 1.4 Where Routing Happens

There is **no ML-based router** in production today. The inference path is:

1. Complexity classifier (TF-IDF + Logistic Regression pickled artifact)
2. Clarity inference (exact-match lookup → heuristic)
3. Tiered benchmark slice (exact → complexity → use_case)
4. Statistical ranking (mean accuracy shortlist → normalized cost/latency value score)
5. Switching policy (hardcoded delta thresholds)

No KNN. No embeddings. No ML ranker. No feedback loop from inference back to training.

---

### 1.5 Embedding / Similarity

**Not present.** The current system performs **categorical slice matching** — filtering rows by exact equality of `use_case`, `prompt_complexity`, and `clarity`. It is not computing semantic similarity between the incoming prompt and benchmark prompts.

---

## PHASE 2 — ARCHITECTURE DOC ANALYSIS

### 2.1 v1 Summary

- 8-stage flow: Ingest → Analyze → Embed → KNN → Rules → ML Ranker → Fusion → Response
- KNN as baseline, rules as hard constraints, ML ranker as refinement
- Scoring: `α·accuracy + β·(1-cost) + γ·(1-latency) + δ·ranker_score`
- Accuracy prediction tiers: KNN weighted → priors → regression
- Edge cases: cold start, new model, OOD prompts
- Deployment: Kubernetes + Kafka + Redis + pgvector

### 2.2 v2 Summary

- 6-stage flow (parallel where possible): Analyze → Embed → Search → Rules → ML → Response
- pgvector HNSW index, K=50, similarity threshold ≥ 0.75
- ML router: LightGBM with confidence gate (≥ 0.7 to override KNN)
- Signal aggregation per model: avg_accuracy, p50_cost, p50_latency, variance, recency_weight
- Conflicting scores: outlier removal, flag rows
- Learning loop: Accept=+1.0, Reject=-0.8
- Drift detection: MMD (data drift), acceptance rate (performance drift)
- Adds `eval_conflict_flag`, `prompt_hash`, `feedback_label` schema fields

### 2.3 Comparison: v1 vs v2

| Dimension | v1 | v2 |
|---|---|---|
| Stage count | 8 (sequential implied) | 6 (parallel stages 1+2, 4+5) |
| KNN vs ML priority | KNN baseline, ML refines | ML first (if confident), KNN fallback |
| Similarity threshold | Not specified | ≥ 0.75 |
| Accuracy prediction | 3-tier (KNN → priors → regression) | 3-strategy (similarity-weighted → regression → CI) |
| Conflict handling | Not addressed | Outlier removal + conflict flag |
| Learning signals | 3 types (user, automated, implicit) | Binary (accept/reject) |
| Drift detection | Not present | MMD + acceptance rate |
| Schema additions | benchmark_prompts + embeddings + models tables | eval_conflict_flag, prompt_hash, feedback_label |
| Deployment model | Kubernetes + Kafka | Not specified |

### 2.4 What Is Actually Useful vs. Over-Engineered

**Useful from v1:**
- 3-tier accuracy prediction (KNN → priors → regression) — maps cleanly to current data
- Cold start via global priors — needed immediately
- The layered decision logic concept (rules gate, then ranking)

**Useful from v2:**
- Parallel execution of prompt analysis + embedding
- Similarity-weighted accuracy aggregation formula
- `eval_conflict_flag` schema field — directly solves current data quality gap
- Confidence intervals for low-data models
- HNSW index for production-scale KNN

**Over-engineered for current stage:**
- v1: Kafka, Kubernetes, Redis — premature infrastructure
- v1: "Mixture of Routers", bandit-based, RL routing — needs years of data
- v2: LightGBM ranker — requires labeled routing outcomes (not benchmark scores), which don't exist yet
- v2: MMD drift detection — valid long-term, but not useful until you have a baseline
- v2: K=50 similarity threshold — needs testing; current dataset may be too sparse for strict cutoffs

**Neither doc addresses the central problem:** the `accuracy_score` → `avg_accuracy_score` migration, evaluator disagreement, and what signal to trust. Both assume clean, reliable scores as input.

---

## PHASE 3 — DATA QUALITY & EVALUATION ANALYSIS

### 3.1 The Current Aggregation Strategy

```python
# From generate_avg_accuracy_scores.py
avg_accuracy_score = mean([
    accuracy_score  (if != 50),   # legacy score, one Gemini evaluator
    eval_llama4_maverick_score,
    eval_mistral_large_score,
    eval_nova_premier_score,
])
```

**Problems with this:**

1. **The legacy `accuracy_score` is not comparable.** It was scored by a single Gemini evaluator using a different (older) prompt structure. Including it in the average is mixing apples with oranges.

2. **Simple averaging treats all evaluators equally.** There is no evidence-based weighting. `llama4-maverick` and `nova-premier` may be systematically biased in opposite directions.

3. **The placeholder guard (≠ 50) is fragile.** A legitimately mediocre response that actually scores 50 will be filtered out. The placeholder guard should be based on `NULL` status of the evaluator column, not value comparison.

4. **No conflict detection.** If `llama4-maverick` gives 90 and `nova-premier` gives 40, the average (65) is meaningless and dangerous. No row is flagged.

5. **Evaluator defaults to score 0 on full failure** (`evaluate_all_responses` in `evaluator.py` line ~312). That 0 gets included if the batch fails silently, dragging down accurate models.

### 3.2 Evaluator Bias Analysis

The three default evaluators per use case are:
- `llama4-maverick`: Meta's MoE model, strong generalist
- `mistral-large`: Strong instruction follower, known to be verbose-reward biased
- `nova-premier`: Amazon's flagship, likely to reward structure over correctness

These three have different training objectives and will have systematic scoring biases. Without calibration data you cannot know the direction. But you can detect it.

**Signal:** Compute per-evaluator mean score across all rows. If `mistral-large` averages 78 and `llama4-maverick` averages 65, one of them is positionally biased. You need to normalize before averaging.

### 3.3 Conflict Detection

A score set has a conflict when the inter-evaluator range is large relative to the scale.

**Proposed rule:**
```
conflict_flag = True  if  max(scores) - min(scores) >= 25
high_conflict  = True  if  max(scores) - min(scores) >= 40
```

Impact thresholds:
- `conflict_flag = True`: Use trimmed mean (drop min+max), record the flag
- `high_conflict = True`: Mark row as `low_confidence`, exclude from ML ranker training
- `stdev > 15`: Flag for human review queue

### 3.4 Better Aggregation Strategy

**Step 1: Per-evaluator z-score normalization (calibration)**

For each evaluator `e`:
```
z_score_e(row) = (raw_score_e - mean_e) / std_e
```

This removes systematic bias. Then convert back to 0-100 scale.

**Step 2: Weighted trimmed mean**

With N evaluators (N ≥ 3):
1. Drop the single highest and single lowest score
2. Compute weighted mean of remaining scores

Initial weights (to be tuned empirically):
```
llama4-maverick:  0.40  (stronger on reasoning/code, less verbose-reward)
nova-premier:     0.35  (strong on text quality)
mistral-large:    0.25  (known verbose-reward bias, lower weight initially)
```

**Step 3: Confidence-weighted final score**
```
final_score = trimmed_weighted_mean
confidence  = 1.0 - (stdev of scores / 50.0)   # 0.0 to 1.0
```

Rows with `confidence < 0.5` should not be used as primary training signal for ML ranker.

### 3.5 When to Discard Rows

| Condition | Action |
|---|---|
| All evaluator columns NULL | Skip row entirely (unevaluated) |
| Only 1 evaluator scored | Use score but mark `low_confidence=True` |
| `high_conflict=True` (range ≥ 40) | Exclude from ML ranker training; keep for analytics |
| `avg_accuracy_score` computed from legacy `accuracy_score` only | Re-evaluate with new evaluators before using for routing |
| Response was empty (model failed) | Row should never have been saved; retroactively mark `invalid=True` |

### 3.6 Impact on Routing

**On the current slice-based recommender:**
- Uses mean `accuracy_score` per model across slice
- If accuracy_score = placeholder 50 or is noisy, the model summary is wrong
- Models with inflated scores from single evaluators can dominate slices
- Tiered filter fallback makes the problem worse: at "use_case_only" tier, all noise accumulates

**On a future KNN:**
- KNN retrieves similar prompts by embedding distance
- If score labels are noisy, KNN returns accurate neighbor vectors but corrupted score values
- Similarity-weighted accuracy then propagates noise directly into the prediction
- A cluster of high-similarity but noisy rows will produce confidently wrong predictions

**On a future ML ranker:**
- ML ranker learns from (features, outcome) pairs
- If outcome = avg_accuracy_score is noisy, the ranker learns noise
- This is worse than no ranker — it confidently generalizes bad patterns

**Conclusion:** Data quality must be fixed before KNN, and KNN before ML ranker. The migration sequence is non-negotiable.

---

## PHASE 4 — GAP ANALYSIS

### Current System vs. Ideal Router

| Component | Current State | What's Missing |
|---|---|---|
| **Score signal** | `accuracy_score` (legacy, 1 evaluator) used in routing | `avg_accuracy_score` is computed but not used by `recommender.py` |
| **Conflict detection** | None | `eval_conflict_flag`, `score_stdev`, `confidence_level` columns |
| **Evaluator weighting** | Uniform average | Per-evaluator calibration + weighted aggregation |
| **Prompt representation** | Categorical slots (complexity, clarity) | No embeddings, no semantic similarity |
| **KNN** | Not present | Vector store + nearest-neighbor retrieval |
| **ML ranker** | Not present | LightGBM on top of KNN signals |
| **Feedback loop** | None | No path from inference → training improvement |
| **Routing log** | None | No record of what was recommended, what was used |
| **Priors** | Not explicit | Cold-start fallback not formalized |
| **Score normalization** | None | Evaluators not calibrated against each other |
| **Row quality flags** | None | `low_confidence`, `invalid`, `eval_conflict_flag` |
| **Caching** | None | Each recommendation re-loads all Supabase rows |
| **Latency** | Full Supabase fetch on every request | No in-memory cache, no vector index |
| **Data pipeline automation** | Manual CLI for avg_accuracy_scores | No scheduled job, no auto-trigger |

### Critical Missing Paths

1. **The evaluator disconnect:** Training pipeline writes `accuracy_score`. The multi-evaluator script writes `avg_accuracy_score`. The recommender reads `accuracy_score`. These three are not connected in production.

2. **No routing telemetry:** You cannot improve what you cannot measure. There is zero logging of inference decisions.

3. **No feedback ingestion:** If a user switches away from the recommended model, there is no mechanism to capture that signal.

4. **Embeddings are completely absent:** You cannot do KNN without them. And without KNN, you cannot do similarity-weighted accuracy prediction.

---

## PHASE 5 — FINAL ARCHITECTURE DESIGN

### 5.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER REQUEST                               │
│              { prompt, use_case, current_model }                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        API GATEWAY                                  │
│             FastAPI — validation, request ID tagging                │
└──────────┬───────────────────────────────────────┬──────────────────┘
           │                                       │
           ▼                                       ▼
┌──────────────────────┐              ┌────────────────────────────┐
│   PROMPT ANALYZER    │              │    EMBEDDING SERVICE       │
│  complexity (ML)     │              │   sentence-transformers    │
│  clarity (lookup)    │   parallel   │   384/768 dim vector       │
│  token estimate      │              │   SHA-256 cache            │
└──────────┬───────────┘              └───────────────┬────────────┘
           │                                          │
           └──────────────────┬───────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │       KNN SEARCH       │
                 │  pgvector HNSW index   │
                 │  K=20, threshold 0.72  │
                 │  filter: use_case      │
                 └───────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌──────────────┐ ┌───────────┐ ┌────────────────┐
     │  SIGNAL AGG  │ │   RULE    │ │   PRIORS DB    │
     │  per-model:  │ │  ENGINE   │ │  (model,use_   │
     │  sim-wtd acc │ │  budget   │ │  case,complex.)│
     │  p50 cost    │ │  latency  │ │  fallback when │
     │  p50 latency │ │  health   │ │  KNN empty     │
     │  variance    │ │  min_n    │ └────────────────┘
     └──────┬───────┘ └─────┬─────┘
            │               │
            └───────┬───────┘
                    ▼
          ┌──────────────────────┐
          │  SCORE FUSION ENGINE │
          │  accuracy: 0.55      │
          │  cost:     0.25      │
          │  latency:  0.15      │
          │  confidence: 0.05    │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │   SWITCHING POLICY   │
          │  compare vs current  │
          │  apply thresholds    │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐   ──────────────────────────────
          │  RESPONSE BUILDER    │ → routing_log (async, fire+forget)
          │  structured payload  │
          └──────────────────────┘
```

---

### 5.2 Core Services

#### 5.2.1 Prompt Analyzer
**What it does today:** TF-IDF + LogReg classifier for complexity. Heuristic + lookup for clarity.

**What stays:**
- Complexity classifier (keep, works)
- Clarity inference (keep, works)

**What gets added:**
- Token count estimation (already done in evaluator.py, move here)
- Intent tag extraction (structured keywords → tag set for use in priors lookup)
- Prompt hash (SHA-256 of normalized prompt → cache key for embeddings)

**Output:**
```python
{
  "complexity": "mid",
  "complexity_confidence": 0.83,
  "clarity": "CLEAR",
  "clarity_source": "classifier",  # or "prompt_logs_exact" | "heuristic"
  "token_estimate": 42,
  "prompt_hash": "a3f2...",
}
```

---

#### 5.2.2 Embedding Service

**Model choice:** `sentence-transformers/all-MiniLM-L6-v2` (384 dim) to start. Upgradeable to `text-embedding-3-small` (OpenAI) or `textembedding-gecko` (Vertex) when budget allows.

**Caching strategy:**
- Check `prompt_embeddings` table by `prompt_hash`
- If hit: return cached vector (avoids model call, <1ms)
- If miss: compute embedding, store in table, return vector

**Storage:** pgvector column (384 or 768 dim). HNSW index for ANN search.

```python
async def get_embedding(prompt: str, prompt_hash: str) -> list[float]:
    cached = await cache_lookup(prompt_hash)
    if cached:
        return cached
    vector = embed_model.encode(prompt).tolist()
    await cache_store(prompt_hash, vector)
    return vector
```

---

#### 5.2.3 KNN Search

**Index:** pgvector `ivfflat` or `hnsw`. HNSW is preferred for production (better recall at query time).

**Query:**
```sql
SELECT
    br.id,
    br.model_id,
    br.provider,
    br.avg_accuracy_score,
    br.cost,
    br.latency_ms,
    br.eval_conflict_flag,
    br.confidence_level,
    1 - (pe.embedding <=> $query_vector) AS similarity
FROM prompt_embeddings pe
JOIN benchmark_results br ON pe.row_id = br.id
WHERE br.use_case = $use_case
  AND br.avg_accuracy_score IS NOT NULL
  AND br.low_confidence = FALSE
  AND 1 - (pe.embedding <=> $query_vector) >= 0.72
ORDER BY pe.embedding <=> $query_vector
LIMIT 30;
```

**K selection:**
- Default: K = 20
- OOD fallback: if fewer than 10 results above threshold, expand to K=50 and lower threshold to 0.60
- Cold model fallback: if a model has < 5 results in K neighbors, inject global prior for that model

**Per-model aggregation from K neighbors:**
```python
for model_id in unique_models_in_neighbors:
    rows = [r for r in neighbors if r.model_id == model_id]
    sim_weighted_accuracy = sum(r.similarity * r.avg_accuracy_score for r in rows) / sum(r.similarity for r in rows)
    p50_cost    = median([r.cost for r in rows])
    p50_latency = median([r.latency_ms for r in rows])
    variance    = stdev([r.avg_accuracy_score for r in rows])
    sample_n    = len(rows)
```

**Signal validity gate:** Only include a model in ranking if `sample_n >= 3` within neighbors. Models below this get replaced by their global prior.

---

#### 5.2.4 Rule Engine

Hard constraints applied **before** ranking. Any model failing a rule is removed from candidate set entirely.

Rules (ordered):
1. **Budget gate:** `p50_cost > budget_ceiling` → exclude (if budget provided by caller)
2. **Latency SLA:** `p50_latency > latency_sla_ms` → exclude (if SLA provided)
3. **Health check:** model reported unhealthy by upstream health monitor → exclude
4. **Minimum sample gate:** `global_sample_count < MIN_SAMPLES_PER_MODEL (5)` → exclude
5. **Use-case capability gate:** model not in `USE_CASE_MODELS[use_case]` → exclude

The rule engine must never produce an empty candidate set without first trying a fallback. If all models are excluded, disable the strictest rule and retry.

---

#### 5.2.5 Score Fusion Engine

**Final composite score:** (replaces current `pick_best_value_model`)

```python
score = (
    0.55 * normalized_accuracy
  + 0.25 * (1 - normalized_cost)
  + 0.15 * (1 - normalized_latency)
  + 0.05 * confidence_bonus
)
```

Where:
- `normalized_*` = min-max normalization across candidate set
- `confidence_bonus` = `1 - (score_variance / 50)` — rewards models with stable evaluator agreement

**Weight presets** (user-selectable):
| Preset | accuracy | cost | latency | confidence |
|---|---|---|---|---|
| `quality` | 0.70 | 0.15 | 0.10 | 0.05 |
| `cost` | 0.40 | 0.45 | 0.10 | 0.05 |
| `latency` | 0.40 | 0.15 | 0.40 | 0.05 |
| `balanced` (default) | 0.55 | 0.25 | 0.15 | 0.05 |

---

#### 5.2.6 Evaluation Pipeline (CRITICAL)

The offline evaluation pipeline (`generate_avg_accuracy_scores.py`) needs significant hardening:

**Problem 1: Placeholder guard is value-based**
```python
# CURRENT (wrong)
if current_score != placeholder_score:  # == 50 check
    scores_for_average.append(current_score)

# FIXED: check by column NULL status, not by value
eval_columns = ["eval_llama4_maverick_score", "eval_mistral_large_score", ...]
valid_scores = [row[col] for col in eval_columns if row.get(col) is not None]
```

**Problem 2: Legacy `accuracy_score` should be excluded from new avg**

The `accuracy_score` field was computed by a single Gemini evaluator with a different system prompt. It should not be mixed into `avg_accuracy_score`. Phase it out explicitly:
```python
# Do NOT include row["accuracy_score"] in scores_for_average
# avg_accuracy_score = mean(eval_*_score columns only)
```

**Problem 3: No conflict detection**

After collecting per-evaluator scores, compute:
```python
score_range = max(scores) - min(scores)
score_stdev = statistics.stdev(scores) if len(scores) > 1 else 0
eval_conflict_flag = score_range >= 25
high_conflict      = score_range >= 40
low_confidence     = len(valid_scores) < len(evaluators) or high_conflict
```

Write these as columns to the row update payload.

**Problem 4: No per-evaluator calibration**

Before computing `avg_accuracy_score`, optionally apply z-score normalization using pre-computed evaluator statistics (mean and std per evaluator across the full dataset). Store calibration parameters in a small table:

```sql
CREATE TABLE evaluator_calibration (
    evaluator_id TEXT PRIMARY KEY,
    global_mean  FLOAT,
    global_std   FLOAT,
    last_computed_at TIMESTAMPTZ
);
```

Calibration run: weekly batch job, computes statistics across all non-NULL evaluator scores.

**Revised aggregation logic:**

```python
def compute_robust_avg(scores: dict[str, int], calibration: dict) -> tuple[float, float, bool]:
    """
    scores: { "llama4-maverick": 82, "mistral-large": 91, "nova-premier": 75 }
    calibration: { "llama4-maverick": {"mean": 76, "std": 14}, ... }
    Returns: (avg_score, confidence, conflict_flag)
    """
    # Step 1: z-score normalize per evaluator
    normalized = {}
    for model, score in scores.items():
        cal = calibration.get(model)
        if cal and cal["std"] > 0:
            z = (score - cal["mean"]) / cal["std"]
            normalized[model] = 50 + (z * 15)  # rescale to ~0-100
        else:
            normalized[model] = score  # no calibration data yet

    values = list(normalized.values())
    score_range = max(values) - min(values)
    conflict_flag = score_range >= 25

    # Step 2: trimmed mean if N >= 3
    if len(values) >= 3:
        values_sorted = sorted(values)
        trimmed = values_sorted[1:-1]  # drop min and max
    else:
        trimmed = values

    avg = sum(trimmed) / len(trimmed)

    # Step 3: confidence
    std = statistics.stdev(values) if len(values) > 1 else 0
    confidence = max(0.0, 1.0 - std / 50.0)

    return round(avg, 2), round(confidence, 3), conflict_flag
```

---

#### 5.2.7 Data Storage Layer

See Section 5.6 for full schema.

---

### 5.3 Evaluation System Design

#### Aggregation Strategy Summary

| Strategy | When to Use | Notes |
|---|---|---|
| Simple mean | ≤ 2 evaluators, no calibration data | Low-confidence output |
| Trimmed mean | ≥ 3 evaluators | Drop min+max before averaging |
| Calibrated weighted mean | After 2+ weeks of evaluator data | Apply z-score norm first |
| Single evaluator | Evaluator failures | Mark `low_confidence=True` |

#### Conflict Resolution

```
score_range < 15  → No conflict, use trimmed mean, high confidence
score_range 15-24 → Minor conflict, use trimmed mean, medium confidence
score_range 25-39 → Conflict detected, set eval_conflict_flag=True, use trimmed mean
score_range ≥ 40  → High conflict, set high_conflict_flag=True, exclude from ML training,
                    use median instead of mean (more robust), mark low_confidence
```

#### Evaluator Weighting (post-calibration)

Start with uniform weights. After 4+ weeks of data, compute per-evaluator agreement rate with human labels (or cross-evaluator vote agreement) and adjust:

```python
EVALUATOR_WEIGHTS = {
    "llama4-maverick": 0.40,  # calibrate empirically
    "nova-premier":    0.35,
    "mistral-large":   0.25,
}
```

#### Rows to Discard

Never delete rows. Soft-mark with `invalid=True`:
- `response` IS NULL or empty
- `avg_accuracy_score` was computed with N=1 and high_conflict (impossible by definition, but guard anyway)
- Row's `accuracy_score = 0` AND all evaluator columns NULL (silent failure during training)

---

### 5.4 Accuracy Prediction Strategy

**Tier 1 — KNN Similarity-Weighted (primary)**

When ≥ 5 neighbor rows exist for a model:
```
predicted_accuracy = Σ(sim_i × avg_accuracy_score_i) / Σ(sim_i)
```

Only use rows where `low_confidence = False` and `eval_conflict_flag = False`.

**Tier 2 — Global Priors (fallback for sparse models)**

Pre-computed per `(model_id, use_case, complexity)`:
```sql
SELECT model_id, use_case, prompt_complexity,
       AVG(avg_accuracy_score) AS prior_accuracy,
       COUNT(*) AS support_n
FROM benchmark_results
WHERE avg_accuracy_score IS NOT NULL AND low_confidence = FALSE
GROUP BY model_id, use_case, prompt_complexity;
```

Materialized view, refreshed daily.

**Tier 3 — Global model mean (last fallback)**

If model has < 5 rows total: use `AVG(avg_accuracy_score)` across all use cases for that model.

**Confidence intervals:**

```python
if neighbor_count >= 20:
    ci = ±4
elif neighbor_count >= 10:
    ci = ±8
elif neighbor_count >= 5:
    ci = ±12
else:
    ci = ±20  # prior-based, very uncertain
```

Include `confidence_interval` in response payload.

---

### 5.5 Data Flow (Step-by-Step)

#### Request-Time Flow (target state)

```
1. Receive: { prompt, use_case, current_model, [budget], [latency_sla] }

2. Validate inputs → 400 if invalid

3. [PARALLEL]
   3a. Prompt Analyzer:
       - load classifier (cached in memory)
       - predict complexity
       - infer clarity
       - compute token estimate
       - compute prompt_hash

   3b. Embedding Service:
       - lookup prompt_hash in prompt_embeddings cache
       - if miss: compute embedding → store
       - return 384-dim vector

4. KNN Search:
   - query pgvector HNSW index (use_case filter)
   - retrieve K=20 neighbors (similarity ≥ 0.72)
   - if insufficient: expand K + lower threshold

5. Per-model signal aggregation:
   - similarity-weighted accuracy per model
   - p50 cost, p50 latency
   - score variance, sample_count

6. Priors injection:
   - for models with < 3 neighbors: replace KNN accuracy with prior
   - for models not in neighbors at all: inject from priors table

7. Rule Engine:
   - apply budget / latency / health / capability gates
   - if all excluded: relax strictest rule + retry

8. Score Fusion:
   - compute composite score per candidate
   - sort descending

9. Switching Policy:
   - compare top candidate vs current_model stats
   - apply delta thresholds

10. Response Builder:
    - format payload
    - include: complexity, clarity, filter_level, top_candidates,
               confidence_intervals, data_source, warnings

11. [ASYNC, fire+forget]
    - write to routing_log: { request_id, prompt_hash, use_case,
                              complexity, clarity, recommended_model,
                              expected_accuracy, timestamp }
```

---

### 5.6 Storage Design

#### Tables

**`benchmark_results`** (existing + additions)

```sql
-- EXISTING (keep)
id                       UUID PRIMARY KEY
provider                 TEXT
model_id                 TEXT
use_case                 TEXT
prompt                   TEXT
prompt_complexity        TEXT
response                 TEXT
accuracy_score           FLOAT      -- legacy, DO NOT use for routing
avg_accuracy_score       FLOAT      -- NEW primary signal
cost                     FLOAT
tokens                   INTEGER
latency_ms               FLOAT
clarity                  TEXT

-- EXISTING evaluator columns
eval_llama4_maverick_score   FLOAT
eval_mistral_large_score     FLOAT
eval_nova_premier_score      FLOAT
eval_deepseek_r1_score       FLOAT

-- ADD THESE
score_stdev              FLOAT      -- stdev across evaluators
eval_conflict_flag       BOOLEAN DEFAULT FALSE
high_conflict_flag       BOOLEAN DEFAULT FALSE
low_confidence           BOOLEAN DEFAULT FALSE
confidence_level         FLOAT      -- 0.0 to 1.0
invalid                  BOOLEAN DEFAULT FALSE
prompt_hash              TEXT       -- SHA-256 of normalized prompt
eval_count               INTEGER    -- how many evaluators contributed
model_version            TEXT       -- model version at time of eval
created_at               TIMESTAMPTZ DEFAULT now()
```

**`prompt_embeddings`** (new)

```sql
CREATE TABLE prompt_embeddings (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_hash  TEXT UNIQUE NOT NULL,
    embedding    vector(384),   -- pgvector
    created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON prompt_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX ON prompt_embeddings (prompt_hash);
```

**`model_priors`** (new — materialized view or table)

```sql
CREATE TABLE model_priors (
    model_id         TEXT,
    use_case         TEXT,
    prompt_complexity TEXT,
    prior_accuracy   FLOAT,
    prior_cost       FLOAT,
    prior_latency    FLOAT,
    support_n        INTEGER,
    last_updated     TIMESTAMPTZ,
    PRIMARY KEY (model_id, use_case, prompt_complexity)
);
```

**`routing_log`** (new)

```sql
CREATE TABLE routing_log (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id         TEXT,
    prompt_hash        TEXT,
    use_case           TEXT,
    complexity         TEXT,
    clarity            TEXT,
    recommended_model  TEXT,
    confidence         FLOAT,
    expected_accuracy  FLOAT,
    data_source        TEXT,     -- 'knn' | 'priors' | 'slice'
    filter_level       TEXT,
    knn_neighbors      INTEGER,
    timestamp          TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON routing_log (prompt_hash);
CREATE INDEX ON routing_log (recommended_model);
CREATE INDEX ON routing_log (timestamp DESC);
```

**`evaluator_calibration`** (new)

```sql
CREATE TABLE evaluator_calibration (
    evaluator_id         TEXT PRIMARY KEY,
    global_mean          FLOAT,
    global_std           FLOAT,
    sample_count         INTEGER,
    last_computed_at     TIMESTAMPTZ
);
```

**`feedback`** (new — for future learning loop)

```sql
CREATE TABLE feedback (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    routing_log_id    UUID REFERENCES routing_log(id),
    prompt_hash       TEXT,
    recommended_model TEXT,
    accepted_model    TEXT,     -- what the user actually used
    signal            TEXT,     -- 'accept' | 'reject' | 'switch'
    signal_weight     FLOAT,    -- +1.0, -0.8, -0.5
    created_at        TIMESTAMPTZ DEFAULT now()
);
```

#### Indexes on `benchmark_results`

```sql
-- existing
CREATE INDEX ON benchmark_results (model_id);
CREATE INDEX ON benchmark_results (prompt_complexity);
CREATE INDEX ON benchmark_results (provider);

-- add
CREATE INDEX ON benchmark_results (use_case);
CREATE INDEX ON benchmark_results (prompt_hash);
CREATE INDEX ON benchmark_results (avg_accuracy_score) WHERE avg_accuracy_score IS NOT NULL;
CREATE INDEX ON benchmark_results (low_confidence, eval_conflict_flag);
```

---

### 5.7 Learning Loop

#### Phase A: Passive Telemetry (implement first, zero user friction)

Every recommendation request writes an async log entry to `routing_log`. This is zero extra user work. Gives you:
- Recommendation volume per model
- Distribution of complexity / clarity / use_case
- KNN vs. priors usage rate

#### Phase B: Implicit Feedback (implement second)

If the frontend knows the user started a new training run after getting a recommendation, that is an implicit "accept." Track:
- Recommendation was shown → training was submitted with a different model → "reject"
- Recommendation was shown → training was submitted with same model → "weak accept"

Write to `feedback` table.

#### Phase C: Continuous Dataset Improvement

**Weekly automation:**
1. Run `generate_avg_accuracy_scores.py --force` on rows older than 30 days that have fewer than 3 evaluator scores
2. Re-compute `evaluator_calibration` statistics
3. Re-compute `model_priors` materialized view
4. Flag rows where `eval_conflict_flag = TRUE` for potential re-evaluation

**Classifier retraining (monthly):**
1. Export rows with `low_confidence = FALSE` from Supabase
2. Retrain TF-IDF + LogReg complexity classifier
3. Evaluate CV accuracy (target > 78%)
4. Replace `classifier.pkl` if improvement ≥ 2% CV accuracy

#### Phase D: ML Ranker (do NOT implement until Phase C is stable)

Only after:
- ≥ 5,000 rows with clean `avg_accuracy_score`
- ≥ 6 months of `routing_log` data
- ≥ 500 feedback entries

LightGBM ranker features:
- prompt signals (complexity, clarity, token_estimate)
- KNN-aggregated accuracy, cost, latency
- model metadata (provider, model_family, parameter_count)
- evaluator confidence and conflict rate

Training target: `avg_accuracy_score` from clean rows (not feedback labels initially).

---

### 5.8 Migration Plan

#### Step 1: Fix the Score Signal (Week 1–2) — **DO THIS NOW**

**What:**
- Update `recommender.py` and `clean_benchmark_rows()` to use `avg_accuracy_score` where available, fall back to `accuracy_score` only where `avg_accuracy_score IS NULL`
- Run `generate_avg_accuracy_scores.py` against all rows without `avg_accuracy_score`

**Code change in `recommender.py`:**
```python
def clean_benchmark_rows(rows: List[dict]) -> List[dict]:
    for row in rows:
        # Use avg_accuracy_score if available and not a placeholder;
        # fall back to accuracy_score only when necessary
        avg = row.get("avg_accuracy_score")
        legacy = row.get("accuracy_score")
        accuracy = avg if avg is not None else legacy
        # ... rest of cleaning
```

**Why first:** Everything downstream depends on this. KNN and ML ranker are worthless if the ground truth signal is wrong.

---

#### Step 2: Add Schema Columns + Quality Flags (Week 2–3)

**What:**
- Add `eval_conflict_flag`, `high_conflict_flag`, `low_confidence`, `score_stdev`, `confidence_level`, `prompt_hash` columns to `benchmark_results`
- Update `generate_avg_accuracy_scores.py` to compute and write all new fields
- Update `recommender.py` to filter out `low_confidence = TRUE` rows from slice

**Why second:** Before adding embeddings, clean the dataset. Dirty data in → dirty vectors out.

---

#### Step 3: Add Embeddings + KNN (Week 3–5)

**What:**
- Add `pgvector` extension to Supabase
- Create `prompt_embeddings` table + HNSW index
- Batch-compute embeddings for all existing `benchmark_results` rows (one-time job)
- Wire embedding service into `recommender.py` (new path, not replacing existing path yet)
- Run both paths in shadow mode: KNN result vs. slice result logged side-by-side

**Shadow mode:** Do not expose KNN result to frontend yet. Log the comparison. Measure:
- Does KNN agree with slice-based result?
- Does KNN produce recommendations for prompts where slice is empty?

---

#### Step 4: Promote KNN to Primary (Week 5–6)

**What:**
- After shadow mode shows KNN ≥ 80% agreement with slice on clean data
- Switch router to KNN-primary, slice-fallback
- Add `model_priors` table for cold-start cases
- Add `routing_log` writes

**Validation metrics:**
- KNN recommendation latency must stay < 200ms P95
- Fallback to priors must be < 5% of requests (indicates enough data)

---

#### Step 5: ML Ranker (Future — 3–6 months out)

Do not commit to this until the data quality and KNN layers are stable. This step requires labeled outcome data, not just scores.

---

### 5.9 Trade-offs

#### Simplicity vs. Accuracy

| Choice | Simple | Accurate |
|---|---|---|
| Routing signal | `accuracy_score` (1 evaluator) | `avg_accuracy_score` (3 evaluators, calibrated) |
| Slice matching | Categorical (current) | KNN embeddings |
| Aggregation | Simple mean | Calibrated trimmed mean |
| Ranking | 2-stage value model (current) | LightGBM ranker |
| Accuracy prediction | Mean of slice | Similarity-weighted KNN |

**Recommendation:** Current categorical slice is good enough for now IF the score signal is fixed. Do not rush to KNN. A fast, reliable categorical recommender with clean data beats a slow, embeddings-based recommender with noisy data.

#### Cost vs. Performance

| Component | Cost | Latency gain |
|---|---|---|
| Local embedding model (MiniLM-L6) | ~$0 (CPU inference) | Adds ~20ms per request |
| Supabase pgvector query | Included in Supabase plan | 5–50ms depending on index warmth |
| Third-party embedding API (OpenAI) | ~$0.02/1M tokens | Adds 100–300ms |
| In-memory embedding cache | RAM only | <1ms for repeat prompts |

**Recommendation:** Use local embedding model (MiniLM-L6-v2 via `sentence-transformers`). Run it as a singleton process in the FastAPI app. Cache embeddings by prompt_hash. Cost remains near zero.

#### Evaluator Cost vs. Score Quality

| Config | Cost per row | Quality |
|---|---|---|
| 2 evaluators | ~$0.003 | Low confidence |
| 3 evaluators (current) | ~$0.005 | Adequate |
| 4 evaluators | ~$0.007 | Better conflict detection |
| 3 evaluators + calibration | ~$0.005 + batch job | Best ROI |

**Recommendation:** Keep 3 evaluators per row. Invest in calibration (batch job, no per-row cost). That improves score quality more than adding a 4th evaluator.

---

## Summary: Priority Order

| Priority | Action | Impact |
|---|---|---|
| 🔴 **P0** | Update `recommender.py` to read `avg_accuracy_score` | Immediately improves routing quality |
| 🔴 **P0** | Run `generate_avg_accuracy_scores.py` on all existing rows | Backfills the signal |
| 🔴 **P0** | Fix placeholder guard (NULL-based, not value-based) | Stops corrupted averages |
| 🟠 **P1** | Add `eval_conflict_flag`, `low_confidence`, `score_stdev` columns | Enables data quality filtering |
| 🟠 **P1** | Update evaluation pipeline to write conflict/confidence fields | Enables P1 above |
| 🟠 **P1** | Add `prompt_hash` to all rows | Required for embedding cache |
| 🟡 **P2** | Deploy pgvector + batch compute embeddings | Enables KNN |
| 🟡 **P2** | Wire embedding service + KNN in shadow mode | Validates before promoting |
| 🟡 **P2** | Build `routing_log` writes | Enables learning loop |
| 🟢 **P3** | Build `model_priors` table | Fixes cold-start |
| 🟢 **P3** | Promote KNN to primary | Better accuracy prediction |
| 🔵 **P4** | Evaluator calibration batch job | Better score aggregation |
| 🔵 **P4** | ML Ranker (LightGBM) | Only after 5K+ clean rows + routing data |

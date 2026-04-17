# LLM Router — System Status, Bug Registry & Learning Loop
### Production Reference Document

---

## 1. CURRENT SYSTEM FLOW

### 1.1 End-to-End Request Flow (Live State)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  USER REQUEST                                                               │
│  POST /api/inference/recommend                                              │
│  { prompt, use_case, current_model }                                        │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  FastAPI            │
                    │  routers/inference  │
                    │  → get_recommendation│
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                                         ▼
┌──────────────────┐                    ┌───────────────────────┐
│ COMPLEXITY       │                    │ CLARITY INFERENCE     │
│ INFERENCE        │                    │                       │
│ classifier.pkl   │                    │ 1. Exact match in     │
│ TF-IDF + LogReg  │                    │    prompt_logs table  │
│ → low/mid/high   │                    │ 2. Exact match in     │
│ → confidence     │   SEQUENTIAL       │    local CSV          │
│ → source         │   (not parallel)   │ 3. Heuristic fallback │
└────────┬─────────┘                    └───────────┬───────────┘
         │                                          │
         └──────────────────┬───────────────────────┘
                            │
                 ┌──────────▼──────────┐
                 │  EMBEDDING SERVICE  │
                 │                     │
                 │ 1. compute_hash()   │
                 │ 2. Supabase cache   │
                 │    lookup           │
                 │ 3. MISS → OpenAI   │
                 │    text-embedding  │
                 │    -3-small (384d)  │
                 │ 4. Upsert cache     │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │   KNN SEARCH        │
                 │   (Primary Path)    │
                 │                     │
                 │ Supabase RPC:       │
                 │ knn_search()        │
                 │ pgvector HNSW       │
                 │ K=20, sim≥0.72      │
                 │                     │
                 │ If <5 neighbors:    │
                 │ retry K=40, 0.60    │
                 └──────────┬──────────┘
                            │
              ┌─────────────┴─────────────┐
              │ ≥5 neighbors              │ <5 neighbors
              ▼                           ▼
┌─────────────────────────┐   ┌──────────────────────────┐
│  KNN SIGNAL AGG         │   │  SLICE FALLBACK          │
│                         │   │  build_slice_recommendation│
│  Per model (≥3 rows):  │   │                          │
│  sim_weighted_accuracy  │   │  Load ALL benchmark rows │
│  p50_cost               │   │  Filter: exact → complex │
│  p50_latency            │   │  → use_case_only         │
│  score_variance         │   │  Summarize + rank        │
│  sample_n               │   │  (legacy fallback path)  │
└────────────┬────────────┘   └──────────────────────────┘
             │
┌────────────▼────────────┐
│  SCORE FUSION           │
│                         │
│  score =                │
│   0.55 × acc_norm       │
│ + 0.25 × (1-cost_norm)  │
│ + 0.15 × (1-lat_norm)   │
│ + 0.05 × confidence     │
│                         │
│  Sort descending → best │
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│  SWITCHING POLICY       │
│                         │
│  Compare best vs        │
│  current_model stats    │
│  Apply thresholds:      │
│  min_accuracy_gain      │
│  max_cost_increase_pct  │
│  max_latency_increase   │
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│  RESPONSE BUILDER       │
│  + async routing_log    │
│    write (fire+forget)  │
└─────────────────────────┘
```

### 1.2 Worked Example (Actual Production Response)

```
Input:
  prompt:        "Write a Python function to merge two sorted lists"
  use_case:      "code-generation"
  current_model: "nova-pro"

Step 1 — Complexity
  classifier.pkl → "low" (confidence: 0.561)

Step 2 — Clarity
  No exact match in prompt_logs → heuristic
  has "write" + "function" → "PARTIAL"

Step 3 — Embedding
  hash = 8e06682ae3df3c1d8b023c964ea51067
  Supabase cache → MISS
  OpenAI text-embedding-3-small → 384-dim vector
  Cached for future requests

Step 4 — KNN Search
  pgvector HNSW: 39 neighbors returned (similarity ≥ 0.72)

Step 5 — Signal Aggregation
  gemini-2-0-flash:  sim_weighted_acc=99.38, p50_cost=$0.000631, p50_lat=20745ms
  llama3-3-70b:      sim_weighted_acc=98.23, p50_cost=$0.000472, p50_lat=4892ms
  llama4-maverick:   sim_weighted_acc=97.03, p50_cost=$0.000575, p50_lat=4140ms

Step 6 — Score Fusion
  gemini-2-0-flash:  0.55×1.00 + 0.25×0.81 + 0.15×0.00 + 0.05×0.94 = 0.805
  llama3-3-70b:      0.55×0.52 + 0.25×1.00 + 0.15×0.67 + 0.05×0.91 = 0.692
  → gemini-2-0-flash wins on accuracy (99.38 vs 98.23)

Step 7 — Switch
  nova-pro not in KNN neighbor set → no comparison → switch=true

Output:
  recommended_model: "gemini-2-0-flash"
  expected_accuracy: 99.38
  knn_confidence:    0.942
  knn_neighbors:     39
  data_source:       "knn"
```

### 1.3 Training Pipeline (Data Generation)

```
Frontend (prompt / CSV upload)
  → POST /api/training/run
  → training.py: process_prompts()
      ├── save_prompt_log()          → Supabase:prompt_logs
      ├── call_all_models()          → 14 Bedrock models (parallel)
      ├── call_all_vertex_models()   → 6 Vertex models (parallel)
      ├── evaluate_all_responses()   → Gemini 2.5 Flash evaluator
      │     (single evaluator, writes accuracy_score only)
      └── save_row()                 → Supabase:benchmark_results
                                       (avg_accuracy_score = NULL at this point)

Offline (manual CLI):
  → generate_avg_accuracy_scores.py
      ├── fetch rows where avg_accuracy_score IS NULL
      ├── for each row: 3 evaluators (llama4, mistral, nova-premier)
      ├── compute avg_accuracy_score
      └── update Supabase row
```

---

## 2. BUG REGISTRY

### BUG-001 — `routing_log` table does not exist
**Severity:** 🟠 Medium  
**Symptom:** Every KNN recommendation logs `[ROUTING LOG ERROR] PGRST205` silently.  
**Impact:** No telemetry collected. Cannot measure KNN vs slice agreement. Cannot build learning loop.  
**Root Cause:** Table was never created in Supabase.  
**Fix:** Run in Supabase SQL Editor:

```sql
CREATE TABLE IF NOT EXISTS routing_log (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id        TEXT,
    prompt_hash       TEXT,
    use_case          TEXT,
    complexity        TEXT,
    clarity           TEXT,
    recommended_model TEXT,
    data_source       TEXT,
    knn_neighbors     INTEGER,
    filter_level      TEXT,
    expected_accuracy FLOAT,
    confidence        FLOAT,
    created_at        TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON routing_log (prompt_hash);
CREATE INDEX ON routing_log (created_at DESC);
CREATE INDEX ON routing_log (recommended_model);
```

---

### BUG-002 — Gemini wins despite 4× worse latency
**Severity:** 🟡 Low-Medium (UX impact)  
**Symptom:** `gemini-2-0-flash` recommended with 20,745ms latency when `llama3-3-70b` gives 98.23% accuracy at 4,892ms.  
**Root Cause:** Score fusion weights accuracy too heavily (0.55) relative to latency (0.15). A 1.15 accuracy point gain overrides a 4× latency penalty.  
**Fix in `recommender.py` → `score_and_rank_knn_candidates()`:**

```python
# Current weights (accuracy-heavy):
item["value_score"] = round(
    0.55 * acc_norm + 0.25 * cost_norm + 0.15 * latency_norm + 0.05 * confidence, 4
)

# Better balanced weights:
item["value_score"] = round(
    0.45 * acc_norm + 0.20 * cost_norm + 0.30 * latency_norm + 0.05 * confidence, 4
)
```

Or make this user-configurable via a `priority` field in the request body.

---

### BUG-003 — `sample_count: 3` for all KNN models (sparse neighbors)
**Severity:** 🟡 Low  
**Symptom:** Every model in the KNN result shows `sample_count: 3` — exactly the `MIN_MODEL_NEIGHBORS` threshold.  
**Root Cause:** Only 548 unique prompts are embedded across 8,594 benchmark rows. Each unique prompt has ~15.7 benchmark rows (multiple models per prompt), so KNN neighbors cluster tightly around the same few prompts.  
**Impact:** Similarity-weighted accuracy based on 3 data points is statistically fragile. High variance.  
**Fix:** Lower `MIN_MODEL_NEIGHBORS` from 3 → 2 to allow sparse models a voice, and add a confidence penalty for low-n models:

```python
# knn_search.py
MIN_MODEL_NEIGHBORS = 2           # was 3

# In aggregate_knn_signals():
"confidence_penalty": 1.0 if len(rows) >= 5 else 0.8 if len(rows) >= 3 else 0.6,
```

Long-term fix: add more diverse prompts to the benchmark dataset.

---

### BUG-004 — Clarity and complexity inference are sequential
**Severity:** 🟡 Low (latency)  
**Symptom:** `infer_clarity()` awaits after `infer_complexity()` finishes. Both can run in parallel.  
**Root Cause:** They are called sequentially in `get_recommendation()`.  
**Fix in `recommender.py`:**

```python
# Current (sequential):
complexity, complexity_confidence, complexity_source = infer_complexity(...)
clarity, clarity_source = await infer_clarity(...)

# Fixed (parallel):
(complexity, complexity_confidence, complexity_source), (clarity, clarity_source) = \
    await asyncio.gather(
        asyncio.to_thread(infer_complexity, prompt, use_case, classifier),
        infer_clarity(prompt, use_case),
    )
```

---

### BUG-005 — Training pipeline writes `accuracy_score` only (single evaluator)
**Severity:** 🔴 High (data quality)  
**Symptom:** Every new training row enters Supabase with `avg_accuracy_score = NULL`. It only gets multi-evaluator scores after manually running `generate_avg_accuracy_scores.py`.  
**Root Cause:** `training.py` uses `evaluate_all_responses()` (Gemini, single evaluator) and saves to `accuracy_score`. `avg_accuracy_score` is a separate offline step.  
**Impact:** New rows are invisible to KNN until the offline script runs. KNN accuracy degrades over time as new data gets added but not indexed.  
**Fix:** After saving a row in `training.py`, schedule it for evaluation:

```python
# training.py — after save_row():
asyncio.create_task(
    schedule_multi_eval(row_id=saved_id, use_case=use_case)
)

# New: services/eval_scheduler.py
async def schedule_multi_eval(row_id: str, use_case: str) -> None:
    """
    Calls generate_avg_accuracy_scores for a single row immediately after
    it's saved, instead of waiting for the manual offline batch.
    """
    try:
        from model_training.generate_avg_accuracy_scores import evaluate_row
        # fetch row, evaluate, update
        ...
    except Exception as exc:
        print(f"[EVAL SCHEDULER ERROR] {exc}")
```

---

### BUG-006 — New benchmark rows have no embeddings (not auto-indexed for KNN)
**Severity:** 🔴 High (KNN coverage)  
**Symptom:** After a training run adds new rows to `benchmark_results`, those rows have no entry in `prompt_embeddings`. They are invisible to KNN.  
**Root Cause:** The training pipeline does not call the embedding service. Embeddings are only created on-demand during inference (lazy, per incoming prompt) or via the backfill script.  
**Impact:** KNN coverage stagnates unless prompts are first submitted to `/recommend` OR the backfill is re-run.  
**Fix:** Embed the prompt at training time and write to `prompt_embeddings`:

```python
# training.py — after save_prompt_log():
asyncio.create_task(
    _index_prompt_embedding(prompt=prompt, supabase_client=supabase)
)

async def _index_prompt_embedding(prompt: str, supabase_client) -> None:
    try:
        from services.embedding_service import get_or_compute_embedding
        await get_or_compute_embedding(prompt, supabase_client)
    except Exception as exc:
        print(f"[EMBED INDEX ERROR] {exc}")
```

---

### BUG-007 — `eval_conflict_flag`, `low_confidence`, `score_stdev` columns missing
**Severity:** 🟠 Medium (data quality)  
**Symptom:** `generate_avg_accuracy_scores.py` computes `avg_accuracy_score` but does not write evaluator conflict flags.  
**Root Cause:** Columns don't exist in the schema yet. No conflict detection is implemented.  
**Impact:** Rows where evaluators disagree wildly (e.g., score 90 vs 40) are treated as reliable. KNN retrieves these rows at full weight.  
**Fix:** Run schema migration + update evaluation pipeline (see Section 4.1).

---

### BUG-008 — Model priors not implemented (cold-start gap)
**Severity:** 🟠 Medium  
**Symptom:** If KNN returns < 5 neighbors after both attempts, the system falls all the way back to the slice-based recommender. There is no priors layer between KNN and slice.  
**Root Cause:** `model_priors` table was designed in the architecture but not implemented.  
**Impact:** For rare/new use cases, the system uses a broad, noisy slice instead of a targeted prior.  
**Fix:** See Section 4.2 (Priors Table).

---

## 3. CURRENT SYSTEM HEALTH SUMMARY

| Metric | Value | Status |
|---|---|---|
| Total benchmark rows | 8,594 | ✅ |
| Rows with `avg_accuracy_score` | 8,593 (99.99%) | ✅ |
| Rows with `prompt_hash` | 8,594 (100%) | ✅ |
| Unique prompts embedded (OpenAI) | 548 | ⚠️ Small corpus |
| `prompt_embeddings` rows | 548 | ⚠️ |
| `routing_log` table | Missing | ❌ |
| `eval_conflict_flag` column | Missing | ❌ |
| `model_priors` table | Missing | ❌ |
| KNN primary path working | Yes | ✅ |
| KNN confidence (latest) | 0.942 | ✅ |
| Slice fallback available | Yes | ✅ |

---

## 4. LEARNING LOOP — PRODUCTION DESIGN

The learning loop closes the gap between recommendations made and outcomes observed.
It has three phases that must be built in order.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LEARNING LOOP OVERVIEW                              │
│                                                                             │
│  INFERENCE PATH                          LEARNING PATH                      │
│                                                                             │
│  Prompt → KNN → Recommend ──────────────→ routing_log                      │
│                     │                         │                            │
│                     │     User uses recommended model                       │
│                     │     OR switches to different model                   │
│                     ▼                         ▼                            │
│              Frontend action ───────────→ feedback table                   │
│                                               │                            │
│                                    ┌──────────▼──────────┐                │
│                                    │  Signal Aggregation  │                │
│                                    │  accept / reject /   │                │
│                                    │  implicit            │                │
│                                    └──────────┬──────────┘                │
│                                               │                            │
│                          ┌────────────────────┼────────────────────┐      │
│                          ▼                    ▼                    ▼      │
│                   avg_accuracy_score    classifier.pkl         model_priors│
│                   weight adjustment      retraining            recompute   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 4.1 Phase A — Telemetry (Implement First, Zero User Friction)

**Create `routing_log` table** (see BUG-001 SQL above).

Every KNN recommendation already calls `_write_routing_log()`. Once the table exists, you get:
- Volume per model recommended
- KNN vs. slice distribution
- Confidence distribution
- Use-case / complexity distribution

No user action required. Passive data collection.

**What to monitor weekly:**
```sql
-- Recommendation distribution
SELECT recommended_model, data_source, COUNT(*) AS requests
FROM routing_log
GROUP BY recommended_model, data_source
ORDER BY requests DESC;

-- KNN health
SELECT
    AVG(knn_neighbors) AS avg_neighbors,
    MIN(knn_neighbors) AS min_neighbors,
    COUNT(CASE WHEN data_source = 'knn' THEN 1 END) AS knn_count,
    COUNT(CASE WHEN data_source != 'knn' THEN 1 END) AS fallback_count
FROM routing_log
WHERE created_at > NOW() - INTERVAL '7 days';

-- Confidence trend
SELECT
    DATE_TRUNC('day', created_at) AS day,
    AVG(confidence) AS avg_confidence,
    COUNT(*) AS requests
FROM routing_log
GROUP BY day
ORDER BY day DESC;
```

---

### 4.2 Phase B — Feedback Capture (Implement Second)

**New Supabase table:**

```sql
CREATE TABLE IF NOT EXISTS feedback (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    routing_log_id    UUID,            -- links to routing_log.id
    prompt_hash       TEXT,
    use_case          TEXT,
    recommended_model TEXT,
    accepted_model    TEXT,            -- what the user actually ran
    signal            TEXT,            -- 'accept' | 'reject' | 'implicit_accept'
    signal_weight     FLOAT,           -- +1.0, -0.8, -0.5
    source            TEXT,            -- 'explicit' | 'implicit'
    created_at        TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON feedback (prompt_hash);
CREATE INDEX ON feedback (recommended_model);
CREATE INDEX ON feedback (signal);
CREATE INDEX ON feedback (created_at DESC);
```

**Signal types and weights:**

| Event | Signal | Weight | How to Capture |
|---|---|---|---|
| User runs recommended model | `accept` | +1.0 | Frontend: training run started with recommended model |
| User runs different model | `reject` | -0.8 | Frontend: training run started with a different model |
| User views recommendation, runs nothing | `implicit_ignore` | -0.2 | Frontend: session end without training run |
| Recommended model returns error | `error` | -0.5 | Backend: model call failure after recommendation |

**Frontend integration (3 events to capture):**

```javascript
// Event 1: Recommendation was shown
const trackRecommendationShown = async (routingLogId, recommendedModel) => {
  sessionStorage.setItem('last_routing_log_id', routingLogId);
  sessionStorage.setItem('last_recommended_model', recommendedModel);
};

// Event 2: Training run submitted
const trackTrainingRun = async (selectedModel) => {
  const routingLogId  = sessionStorage.getItem('last_routing_log_id');
  const recommendedModel = sessionStorage.getItem('last_recommended_model');
  if (!routingLogId) return;

  const signal = selectedModel === recommendedModel ? 'accept' : 'reject';
  await fetch('/api/feedback/submit', {
    method: 'POST',
    body: JSON.stringify({
      routing_log_id:    routingLogId,
      accepted_model:    selectedModel,
      signal,
      signal_weight:     signal === 'accept' ? 1.0 : -0.8,
    })
  });
  sessionStorage.removeItem('last_routing_log_id');
};
```

**New backend endpoint:**

```python
# routers/feedback.py
from fastapi import APIRouter
from services.supabase_client import supabase

router = APIRouter(prefix="/api/feedback")

@router.post("/submit")
async def submit_feedback(body: FeedbackRequest):
    supabase.table("feedback").insert({
        "routing_log_id":    body.routing_log_id,
        "recommended_model": body.recommended_model,
        "accepted_model":    body.accepted_model,
        "signal":            body.signal,
        "signal_weight":     body.signal_weight,
        "source":            "explicit",
    }).execute()
    return {"status": "ok"}
```

---

### 4.3 Phase C — Data Quality Hardening

**Schema additions to `benchmark_results`:**

```sql
ALTER TABLE benchmark_results
    ADD COLUMN IF NOT EXISTS score_stdev        FLOAT,
    ADD COLUMN IF NOT EXISTS eval_conflict_flag  BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS high_conflict_flag  BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS low_confidence      BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS confidence_level    FLOAT,
    ADD COLUMN IF NOT EXISTS eval_count          INTEGER;

CREATE INDEX ON benchmark_results (low_confidence, eval_conflict_flag)
    WHERE avg_accuracy_score IS NOT NULL;
```

**Update `generate_avg_accuracy_scores.py`** to write quality flags:

```python
def compute_quality_flags(scores: list[float]) -> dict:
    if len(scores) < 2:
        return {
            "score_stdev":        None,
            "eval_conflict_flag": False,
            "high_conflict_flag": False,
            "low_confidence":     True,
            "confidence_level":   0.3,
            "eval_count":         len(scores),
        }
    score_range = max(scores) - min(scores)
    stdev       = statistics.stdev(scores)
    return {
        "score_stdev":        round(stdev, 3),
        "eval_conflict_flag": score_range >= 25,
        "high_conflict_flag": score_range >= 40,
        "low_confidence":     score_range >= 40 or len(scores) < 2,
        "confidence_level":   round(max(0.0, 1.0 - stdev / 50.0), 3),
        "eval_count":         len(scores),
    }
```

**Update KNN SQL function** to exclude low-confidence rows:

```sql
-- Add to WHERE clause in knn_search():
AND COALESCE(br.low_confidence, FALSE) = FALSE
AND COALESCE(br.eval_conflict_flag, FALSE) = FALSE
```

---

### 4.4 Phase D — Model Priors (Cold-Start Fix)

**New table:**

```sql
CREATE TABLE IF NOT EXISTS model_priors (
    model_id          TEXT,
    use_case          TEXT,
    prompt_complexity TEXT,
    prior_accuracy    FLOAT,
    prior_cost        FLOAT,
    prior_latency     FLOAT,
    support_n         INTEGER,
    last_updated      TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (model_id, use_case, prompt_complexity)
);
```

**Weekly refresh job:**

```sql
INSERT INTO model_priors
    (model_id, use_case, prompt_complexity,
     prior_accuracy, prior_cost, prior_latency, support_n, last_updated)
SELECT
    model_id, use_case, prompt_complexity,
    AVG(avg_accuracy_score),
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cost),
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms),
    COUNT(*),
    now()
FROM benchmark_results
WHERE avg_accuracy_score IS NOT NULL
  AND COALESCE(low_confidence, FALSE) = FALSE
GROUP BY model_id, use_case, prompt_complexity
ON CONFLICT (model_id, use_case, prompt_complexity)
DO UPDATE SET
    prior_accuracy = EXCLUDED.prior_accuracy,
    prior_cost     = EXCLUDED.prior_cost,
    prior_latency  = EXCLUDED.prior_latency,
    support_n      = EXCLUDED.support_n,
    last_updated   = now();
```

**Use in recommender:** When KNN returns < 3 rows for a model, inject its prior instead of excluding it from ranking.

---

### 4.5 Phase E — Automated Retraining

> **Prerequisite:** Phases A, B, C must be stable. Target: 500+ feedback entries.

#### Complexity Classifier Retraining

The `classifier.pkl` (TF-IDF + Logistic Regression) is static. It never improves.

**Retraining pipeline** (monthly, or when CV accuracy drops below 78%):

```python
# model_training/retrain_classifier.py
"""
Retrains the complexity classifier using labeled benchmark rows.
Labels: prompt_complexity column (low/mid/high, human-assigned at upload time).
"""
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pickle

def retrain(rows: list[dict]) -> None:
    X = [
        f"use_case: {r['use_case']}\nprompt: {r['prompt']}"
        for r in rows
        if r.get('prompt') and r.get('prompt_complexity')
    ]
    y = [r['prompt_complexity'] for r in rows if r.get('prompt_complexity')]

    clf = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=20000)),
        ('lr',    LogisticRegression(max_iter=1000, C=1.0)),
    ])

    # Cross-validate before committing
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    if cv_scores.mean() < 0.78:
        print("CV accuracy below threshold — not replacing classifier.pkl")
        return

    clf.fit(X, y)
    with open("artifacts/classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    print(f"✓ Saved new classifier (CV: {cv_scores.mean():.3f})")
```

**Trigger conditions:**
- Monthly scheduled run
- When `routing_log` shows complexity distribution shift (>20% change in low/mid/high ratio week-over-week)
- When user explicitly flags a complexity prediction as wrong (future UI feature)

#### Score Weight Adjustment (Feedback-Driven)

Use accumulated feedback to tune score fusion weights:

```python
# model_training/tune_weights.py
"""
Compute optimal score fusion weights based on feedback outcomes.
For each accepted recommendation, the weights that produced it should
be reinforced. For rejected ones, they should be penalized.
"""

def compute_weight_gradient(feedback_rows: list[dict]) -> dict:
    """
    Simple gradient: for each accepted row, the recommended model
    had the highest composite score. Compute which weight vector
    best predicts accepts vs rejects.
    """
    accepts = [r for r in feedback_rows if r['signal'] == 'accept']
    rejects = [r for r in feedback_rows if r['signal'] == 'reject']

    accept_rate = len(accepts) / max(len(feedback_rows), 1)
    print(f"Accept rate: {accept_rate:.2%}")

    # Heuristic: if latency of recommended model > 10s AND rejection rate > 40%
    # → increase latency weight
    high_latency_rejects = [
        r for r in rejects
        if r.get('recommended_latency_ms', 0) > 10000
    ]
    if len(high_latency_rejects) / max(len(rejects), 1) > 0.4:
        print("→ High latency is causing rejections. Suggest increasing latency weight.")
        return {
            "accuracy": 0.45,
            "cost":     0.20,
            "latency":  0.30,
            "confidence": 0.05,
        }

    return {
        "accuracy": 0.55,
        "cost":     0.25,
        "latency":  0.15,
        "confidence": 0.05,
    }
```

---

### 4.6 Phase F — Drift Detection (Long-Term)

> **Prerequisite:** 6+ months of `routing_log` data.

**Two drift signals to monitor:**

#### 1. Acceptance Rate Drift (Performance Drift)

```sql
-- Week-over-week acceptance rate
SELECT
    DATE_TRUNC('week', f.created_at)  AS week,
    COUNT(CASE WHEN f.signal = 'accept' THEN 1 END)::float
        / COUNT(*)                    AS acceptance_rate,
    COUNT(*)                          AS total_feedback
FROM feedback f
GROUP BY week
ORDER BY week DESC;
```

Alert if acceptance rate drops below 60% in any week.

#### 2. Prompt Distribution Drift (Data Drift)

Track the centroid of incoming prompt embeddings over time. If the centroid shifts significantly, the benchmark corpus no longer represents the incoming prompt distribution.

```python
# Weekly batch job
async def check_embedding_drift():
    # Get average embedding of last week's prompts from routing_log
    # Compare to previous week's average embedding
    # If cosine similarity < 0.85 → flag drift
    ...
```

---

## 5. IMPLEMENTATION PRIORITY

```
┌────────────────────────────────────────────────────────────────────────────┐
│  PRIORITY    │ ACTION                           │ TIME  │ IMPACT           │
├──────────────┼──────────────────────────────────┼───────┼──────────────────┤
│ 🔴 P0 NOW   │ Create routing_log table (BUG-001)│ 5min  │ Telemetry ON     │
│ 🔴 P0 NOW   │ Add embed call in training.py     │ 1hr   │ KNN auto-indexed │
│              │ (BUG-006)                         │       │                  │
├──────────────┼──────────────────────────────────┼───────┼──────────────────┤
│ 🟠 P1       │ Rebalance score weights (BUG-002) │ 10min │ Better latency   │
│ 🟠 P1       │ Add feedback table + endpoint     │ 1 day │ Signal capture   │
│ 🟠 P1       │ Frontend feedback events          │ 1 day │ Accept/reject    │
│ 🟠 P1       │ Add eval_conflict_flag schema     │ 2hr   │ Data quality     │
├──────────────┼──────────────────────────────────┼───────┼──────────────────┤
│ 🟡 P2       │ model_priors table + weekly job   │ 1 day │ Cold-start fix   │
│ 🟡 P2       │ Update knn_search SQL to filter  │ 30min │ Cleaner KNN      │
│              │ low_confidence rows               │       │                  │
│ 🟡 P2       │ Parallel complexity + clarity     │ 30min │ -50ms latency    │
│              │ (BUG-004)                         │       │                  │
├──────────────┼──────────────────────────────────┼───────┼──────────────────┤
│ 🟢 P3       │ Classifier retraining pipeline    │ 1 day │ Better classify  │
│ 🟢 P3       │ Feedback-driven weight tuning     │ 2 days│ Auto-improve     │
├──────────────┼──────────────────────────────────┼───────┼──────────────────┤
│ 🔵 P4       │ Drift detection batch jobs        │ 1 wk  │ Long-term health │
│ 🔵 P4       │ LightGBM ranker                   │ 2 wks │ After 5K feeback │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. KEY METRICS TO TRACK

| Metric | Target | Alarm If |
|---|---|---|
| KNN usage rate (`data_source=knn`) | > 85% | < 70% |
| KNN avg neighbors (`knn_neighbors`) | > 15 | < 8 |
| KNN avg confidence | > 0.80 | < 0.65 |
| Feedback acceptance rate | > 65% | < 50% |
| Rows with `avg_accuracy_score` | > 98% | < 90% |
| `prompt_embeddings` coverage | = unique prompt_hashes | Diverges |
| Classifier CV accuracy (monthly) | > 78% | < 75% |
| Routing P95 latency | < 800ms | > 1500ms |

# LLM Router — Implementation Plan

## How a Recommendation Works (Read This First)

Before the plan, understand the **full journey** of a single recommendation request,
both today and in the target state.

---

### THE JOURNEY: User picks a use case + types a prompt

**Input from the user:**
```
use_case:      "code-generation"
prompt:        "Write a Python function that merges two sorted lists"
current_model: "nova-pro"
```

---

### TODAY'S FLOW (Current Code)

```
Step 1 — Complexity inference
   The backend loads classifier.pkl (TF-IDF + Logistic Regression)
   It builds: "use_case: code-generation\nprompt: Write a Python function..."
   It predicts: complexity = "mid"  (confidence: 0.81)
   Source: "classifier"

Step 2 — Clarity inference
   Look up this exact prompt in Supabase prompt_logs table
   → not found (new prompt)
   Fall back to heuristics:
   → has "write" (explicit verb) ✓
   → has "function" (constraint) ✓
   → word count = 10 (>= 8) ✓
   → clarity = "CLEAR"
   Source: "heuristic"

Step 3 — Load benchmark data
   Call Supabase:
     SELECT * FROM benchmark_results WHERE use_case = 'code-generation'
   Get back ~8,000 rows

Step 4 — Build the slice (Tiered filter)
   Try EXACT:  use_case='code-generation' AND complexity='mid' AND clarity='CLEAR'
   → 2,100 rows from 9 models
   → All 9 models have >= 5 samples ✓
   → Use this slice (filter_level = "exact")

Step 5 — Summarize models
   For each model compute across those 2,100 rows:
   ┌──────────────────┬──────────────┬────────────┬───────────────┬──────────┐
   │ model_id         │ avg_accuracy │ median_cost│ median_latency│ sample_n │
   ├──────────────────┼──────────────┼────────────┼───────────────┼──────────┤
   │ gemini-2-5-pro   │    87.4      │  $0.000312 │   2,100 ms    │   340    │
   │ nova-premier     │    85.1      │  $0.000891 │   3,200 ms    │   290    │
   │ deepseek-r1      │    84.8      │  $0.000421 │   4,800 ms    │   210    │
   │ llama4-maverick  │    84.2      │  $0.000118 │   1,900 ms    │   280    │
   │ nova-pro         │    81.3      │  $0.000203 │   2,400 ms    │   260    │
   └──────────────────┴──────────────┴────────────┴───────────────┴──────────┘

Step 6 — Quality shortlist
   top_accuracy = 87.4
   ACCURACY_TOLERANCE = 2.0
   shortlist = models where avg_accuracy >= 87.4 - 2.0 = 85.4
   → shortlist: [gemini-2-5-pro (87.4), nova-premier (85.1)]
   Wait — nova-premier is 85.1 < 85.4, so it's excluded.
   shortlist = [gemini-2-5-pro]
   Only 1 model makes the cut.

Step 7 — Value score within shortlist
   Only gemini-2-5-pro qualifies → it wins automatically

Step 8 — Switching policy
   recommended = gemini-2-5-pro  (acc=87.4)
   current     = nova-pro        (acc=81.3)
   accuracy_gain = 87.4 - 81.3 = 6.1  >=  MIN_ACCURACY_GAIN (2.0) ✓
   → switch_recommended = True
   → reason: "Accuracy gain is large enough to justify switching."

Output:
   recommended_model: "gemini-2-5-pro"
   expected_accuracy: 87.4
   expected_cost:     $0.000312
   switch_recommended: true
   reason: "Switch from nova-pro to gemini-2-5-pro. Accuracy gain is large enough..."
```

**What's BROKEN about this today:**
- Step 3 returns rows. The `clean_benchmark_rows()` function reads `accuracy_score`
  (1 evaluator, legacy). NOT `avg_accuracy_score` (3 evaluators, multi-eval).
- So Step 5's numbers are based on noisy, single-evaluator scores.
- If the right model got unlucky with a single bad Gemini evaluation, it might
  lose to a worse model that got lucky.

---

### TARGET FLOW (After Full Implementation)

```
Step 1 — Prompt Analyzer [same as today, already works]
   complexity = "mid", clarity = "CLEAR"

Step 2 — Embedding Service [NEW]
   Compute SHA-256 hash of normalized prompt
   prompt_hash = "a3f29d..."
   Look up in prompt_embeddings table → MISS (first time)
   Run: sentence-transformers/all-MiniLM-L6-v2
   Output: 384-dimensional vector [0.021, -0.143, 0.892, ...]
   Store in prompt_embeddings table
   ★ Steps 1 and 2 run IN PARALLEL

Step 3 — KNN Search [NEW]
   Query pgvector:
     Find 20 most similar prompt-vectors in benchmark_results
     WHERE use_case = 'code-generation'
       AND avg_accuracy_score IS NOT NULL
       AND low_confidence = FALSE
       AND similarity >= 0.72
   Returns 20 neighbor rows, each with a similarity score.

   Example neighbors returned:
   ┌────────────────────────────────────────┬──────────┬─────────────────┬────────────┐
   │ prompt (truncated)                     │similarity│ model_id        │ avg_acc    │
   ├────────────────────────────────────────┼──────────┼─────────────────┼────────────┤
   │ "Write a function to merge two arrays" │  0.94    │ gemini-2-5-pro  │   91.0     │
   │ "Write a function to merge two arrays" │  0.94    │ llama4-maverick │   87.0     │
   │ "Merge k sorted linked lists in Python"│  0.81    │ gemini-2-5-pro  │   88.5     │
   │ "Write Python code to interleave lists"│  0.78    │ nova-premier    │   86.0     │
   │ "Implement merge sort in Python"       │  0.74    │ deepseek-r1     │   85.0     │
   │ ... 15 more rows ...                   │          │                 │            │
   └────────────────────────────────────────┴──────────┴─────────────────┴────────────┘

Step 4 — Per-model signal aggregation [NEW]
   For gemini-2-5-pro (8 neighbor rows):
     sim_weighted_accuracy = (0.94×91.0 + 0.81×88.5 + ...) / (0.94+0.81+...)
                           = 89.7
     p50_cost    = $0.000312
     p50_latency = 2,100 ms
     score_variance = 3.2  (low → reliable)

   For llama4-maverick (6 neighbor rows):
     sim_weighted_accuracy = 85.3
     p50_cost    = $0.000118
     p50_latency = 1,900 ms
     score_variance = 5.1

   For nova-premier (4 neighbor rows):  ← only 4, below threshold of 5
     → inject PRIOR from model_priors table instead
     prior_accuracy = 84.5

Step 5 — Rule Engine
   Apply gates:
   ✓ All models in use_case registry
   ✓ No budget constraint passed
   ✓ All models healthy
   ✓ All above MIN_SAMPLES (or using priors)
   → No models excluded

Step 6 — Score Fusion (composite score)
   score = 0.55×acc_norm + 0.25×(1-cost_norm) + 0.15×(1-latency_norm) + 0.05×confidence

   gemini-2-5-pro:  score = 0.92  ← winner
   llama4-maverick: score = 0.81
   nova-premier:    score = 0.76

Step 7 — Switching policy [same logic, better data]
   recommended = gemini-2-5-pro (sim-weighted acc = 89.7)
   current     = nova-pro       (global prior acc = 81.3)
   gain = 8.4 → SWITCH

Output:
   recommended_model:   "gemini-2-5-pro"
   expected_accuracy:   89.7   (±4 CI, from 20 neighbors)
   data_source:         "knn"
   knn_neighbors_used:  20
   confidence:          0.91
   switch_recommended:  true
```

**What's BETTER:**
- Accuracy is similarity-weighted from semantically close prompts, not a
  broad average across ALL code prompts ever run.
- Conflict-flagged rows are excluded. Score is trustworthy.
- Confidence interval tells you how sure the system is.
- Cold-start models get priors injection instead of being silently ignored.

---

## Implementation Plan

---

## PHASE 0 — Fix the Score Signal
### Timeline: 1–2 days | Risk: Low | Impact: Immediate

This is the highest-leverage change in the entire plan.
One function fix. Immediate improvement in routing quality.

---

### P0.1 — Update `recommender.py` to use `avg_accuracy_score`

**File:** `backend/services/recommender.py`

**Current `clean_benchmark_rows()` (lines 198–228):**
```python
accuracy_score = float(row["accuracy_score"])   # ← WRONG
```

**Replace with:**
```python
def clean_benchmark_rows(rows: List[dict]) -> List[dict]:
    cleaned: List[dict] = []
    for row in rows:
        try:
            model_id   = str(row["model_id"]).strip()
            provider   = str(row.get("provider", "")).strip()
            use_case   = str(row["use_case"]).strip().lower()
            complexity = str(row["prompt_complexity"]).strip().lower()
            clarity    = str(row["clarity"]).strip().upper()

            # ★ Use avg_accuracy_score if present and non-null,
            #   fall back to accuracy_score only when necessary
            avg_acc  = row.get("avg_accuracy_score")
            leg_acc  = row.get("accuracy_score")

            if avg_acc is not None:
                try:
                    accuracy_score = float(avg_acc)
                except (TypeError, ValueError):
                    accuracy_score = None
            elif leg_acc is not None:
                try:
                    accuracy_score = float(leg_acc)
                except (TypeError, ValueError):
                    accuracy_score = None
            else:
                continue  # no score at all, skip row

            if accuracy_score is None:
                continue

            cost       = float(row["cost"])
            latency_ms = float(row["latency_ms"])

        except (KeyError, TypeError, ValueError):
            continue

        if (not model_id or use_case == ""
                or complexity not in VALID_COMPLEXITIES
                or clarity not in VALID_CLARITIES):
            continue

        cleaned.append({
            "model_id":         model_id,
            "provider":         provider,
            "use_case":         use_case,
            "prompt_complexity": complexity,
            "clarity":          clarity,
            "accuracy_score":   accuracy_score,   # now avg_accuracy_score when available
            "cost":             cost,
            "latency_ms":       latency_ms,
            "has_multi_eval":   avg_acc is not None,  # audit flag
        })
    return cleaned
```

Also update `get_benchmark_data()` in `supabase_client.py` to SELECT
`avg_accuracy_score` explicitly (it currently does `select("*")` so it's
already included, but make it explicit for clarity).

---

### P0.2 — Run the evaluation pipeline on all unscored rows

```bash
cd model_training
python generate_avg_accuracy_scores.py \
    --use-case text-generation \
    --max-workers 6

python generate_avg_accuracy_scores.py \
    --use-case code-generation \
    --max-workers 6

python generate_avg_accuracy_scores.py \
    --use-case reasoning \
    --max-workers 6
```

This populates `avg_accuracy_score` for all rows that are currently NULL.
Run on all three use cases. Do NOT use `--force` yet (don't overwrite existing scores).

---

### P0.3 — Fix the placeholder guard in `generate_avg_accuracy_scores.py`

**File:** `model_training/generate_avg_accuracy_scores.py`

**Current `evaluate_row()` (lines 421–423):**
```python
current_score = _coerce_existing_score(row.get("accuracy_score"))
if current_score is not None and current_score != placeholder_score:
    scores_for_average.append(current_score)
```

**Replace with:**
```python
# Do NOT include legacy accuracy_score in multi-evaluator average.
# It was produced by a different evaluator model with a different system prompt.
# avg_accuracy_score = mean of eval_*_score columns ONLY.
# (Remove the 3 lines above entirely.)
```

And update the average computation:
```python
# Only count scores from named evaluator columns
for evaluator in evaluators:
    try:
        score = evaluate_with_model(...)
        column_name = make_score_column_name(evaluator.short_id)
        update_payload[column_name] = score
        scores_for_average.append(float(score))
        successful_new_scores += 1
    except Exception as exc:
        evaluator_failures.append(f"{evaluator.short_id}: {exc}")

if successful_new_scores == 0:
    raise ValueError(f"All evaluators failed: {' | '.join(evaluator_failures)}")

# avg = mean of ONLY the multi-evaluator scores (not legacy accuracy_score)
update_payload["avg_accuracy_score"] = round(
    sum(scores_for_average) / len(scores_for_average), 2
)
```

---

### P0 Validation

After running, verify in Supabase:
```sql
-- Check coverage
SELECT
    COUNT(*) AS total_rows,
    COUNT(avg_accuracy_score) AS has_avg_score,
    ROUND(100.0 * COUNT(avg_accuracy_score) / COUNT(*), 1) AS pct_covered
FROM benchmark_results;

-- Check no legacy contamination
SELECT AVG(avg_accuracy_score), STDDEV(avg_accuracy_score)
FROM benchmark_results
WHERE avg_accuracy_score IS NOT NULL;
-- Expect: mean 65-85, stdev < 20. If stdev >> 20, something is wrong.

-- Check per-model accuracy looks reasonable
SELECT model_id, COUNT(*), AVG(avg_accuracy_score), STDDEV(avg_accuracy_score)
FROM benchmark_results
WHERE avg_accuracy_score IS NOT NULL
GROUP BY model_id
ORDER BY AVG(avg_accuracy_score) DESC;
```

---

## PHASE 1 — Data Quality Flags + Schema
### Timeline: 3–5 days | Risk: Low | Impact: Cleaner data for KNN

---

### P1.1 — Add new columns to Supabase

Run in Supabase SQL editor:

```sql
ALTER TABLE benchmark_results
    ADD COLUMN IF NOT EXISTS score_stdev       FLOAT,
    ADD COLUMN IF NOT EXISTS eval_conflict_flag BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS high_conflict_flag BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS low_confidence     BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS confidence_level   FLOAT,
    ADD COLUMN IF NOT EXISTS eval_count         INTEGER,
    ADD COLUMN IF NOT EXISTS prompt_hash        TEXT,
    ADD COLUMN IF NOT EXISTS invalid            BOOLEAN DEFAULT FALSE;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_br_prompt_hash
    ON benchmark_results (prompt_hash);

CREATE INDEX IF NOT EXISTS idx_br_low_confidence
    ON benchmark_results (low_confidence, eval_conflict_flag)
    WHERE avg_accuracy_score IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_br_use_case
    ON benchmark_results (use_case);
```

---

### P1.2 — Update evaluation pipeline to write quality flags

**File:** `model_training/generate_avg_accuracy_scores.py`

Add a new function:
```python
import hashlib
import re
import statistics

def compute_prompt_hash(prompt: str) -> str:
    """SHA-256 of lowercased, whitespace-normalized prompt."""
    normalized = re.sub(r"\s+", " ", (prompt or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def compute_quality_flags(scores: list[float]) -> dict:
    """
    Given a list of per-evaluator scores, compute conflict and confidence metrics.
    Returns dict of fields to write back to the row.
    """
    if len(scores) < 2:
        return {
            "score_stdev":        None,
            "eval_conflict_flag": False,
            "high_conflict_flag": False,
            "low_confidence":     True,   # only 1 score = low confidence
            "confidence_level":   0.3,
            "eval_count":         len(scores),
        }

    score_range = max(scores) - min(scores)
    stdev       = statistics.stdev(scores)
    conflict    = score_range >= 25
    high_conf_  = score_range >= 40
    confidence  = max(0.0, round(1.0 - stdev / 50.0, 3))

    return {
        "score_stdev":        round(stdev, 3),
        "eval_conflict_flag": conflict,
        "high_conflict_flag": high_conf_,
        "low_confidence":     high_conf_ or len(scores) < 2,
        "confidence_level":   confidence,
        "eval_count":         len(scores),
    }
```

Then update `evaluate_row()` to call this and include the result in `update_payload`:

```python
def evaluate_row(row, placeholder_score, override_names=None):
    prompt   = str(row.get("prompt", "") or "").strip()
    response = str(row.get("response", "") or "").strip()
    use_case = str(row.get("use_case", "text-generation") or "text-generation").strip().lower()

    evaluators = resolve_evaluators_for_use_case(use_case, override_names=override_names)
    update_payload = {}
    scores_for_average = []
    successful_new_scores = 0
    evaluator_failures = []

    for evaluator in evaluators:
        try:
            score = evaluate_with_model(model=evaluator, use_case=use_case,
                                        prompt=prompt, response=response)
            column_name = make_score_column_name(evaluator.short_id)
            update_payload[column_name] = score
            scores_for_average.append(float(score))
            successful_new_scores += 1
        except Exception as exc:
            evaluator_failures.append(f"{evaluator.short_id}: {exc}")

    if successful_new_scores == 0:
        raise ValueError(f"All evaluators failed: {' | '.join(evaluator_failures)}")

    # Compute avg
    update_payload["avg_accuracy_score"] = round(
        sum(scores_for_average) / len(scores_for_average), 2
    )

    # ★ Compute and write quality flags
    flags = compute_quality_flags(scores_for_average)
    update_payload.update(flags)

    # ★ Write prompt hash
    update_payload["prompt_hash"] = compute_prompt_hash(prompt)

    # Metadata (popped before DB write)
    update_payload["_meta_successful_new_scores"] = successful_new_scores
    update_payload["_meta_evaluator_failures"]    = evaluator_failures
    return update_payload
```

---

### P1.3 — Update `recommender.py` to filter low-confidence rows

In `clean_benchmark_rows()`, add a filter:
```python
# Skip rows explicitly marked as low confidence (only applies once P1.2 is deployed)
if row.get("low_confidence") is True:
    continue
# Skip rows marked invalid
if row.get("invalid") is True:
    continue
```

---

### P1.4 — Backfill prompt_hash on existing rows

One-time batch script:
```python
# model_training/backfill_prompt_hash.py
import hashlib, re, os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(Path(__file__).parent.parent / "backend" / ".env")

def compute_prompt_hash(prompt: str) -> str:
    normalized = re.sub(r"\s+", " ", (prompt or "").strip().lower())
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

start = 0
PAGE = 500
while True:
    rows = (supabase.table("benchmark_results")
            .select("id,prompt")
            .is_("prompt_hash", "null")
            .range(start, start + PAGE - 1)
            .execute().data or [])
    if not rows:
        break
    for row in rows:
        ph = compute_prompt_hash(row["prompt"])
        supabase.table("benchmark_results").update({"prompt_hash": ph}).eq("id", row["id"]).execute()
    start += PAGE
    print(f"Done: {start}")

print("Backfill complete.")
```

---

### P1 Validation

```sql
-- Check flag distribution
SELECT
    eval_conflict_flag,
    high_conflict_flag,
    low_confidence,
    COUNT(*) AS rows
FROM benchmark_results
WHERE avg_accuracy_score IS NOT NULL
GROUP BY 1, 2, 3
ORDER BY 3 DESC, 1 DESC;

-- Expect: most rows should be low_confidence=FALSE, conflict=FALSE
-- If >30% are high_conflict, evaluator calibration is needed urgently

-- Check prompt_hash coverage
SELECT COUNT(*) FROM benchmark_results WHERE prompt_hash IS NULL;
-- Should be 0 after backfill
```

---

## PHASE 2 — Embeddings + KNN (Shadow Mode)
### Timeline: 1–2 weeks | Risk: Medium | Impact: Unlocks semantic routing

---

### P2.1 — Enable pgvector in Supabase

In Supabase dashboard → Database → Extensions → Enable `vector`.

Then create the embeddings table:
```sql
CREATE TABLE IF NOT EXISTS prompt_embeddings (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_hash TEXT UNIQUE NOT NULL,
    embedding   vector(384),
    model_name  TEXT DEFAULT 'all-MiniLM-L6-v2',
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_pe_prompt_hash
    ON prompt_embeddings (prompt_hash);

CREATE INDEX IF NOT EXISTS idx_pe_embedding_hnsw
    ON prompt_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

---

### P2.2 — Add embedding service

**New file:** `backend/services/embedding_service.py`

```python
"""
Embedding service using sentence-transformers (runs locally, zero API cost).
Caches by prompt_hash to avoid redundant computation.
"""
from __future__ import annotations
import hashlib
import re
import os
from functools import lru_cache
from typing import Optional

# Lazy import — only loads model on first call
_model = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def compute_prompt_hash(prompt: str) -> str:
    normalized = re.sub(r"\s+", " ", (prompt or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def embed_text(text: str) -> list[float]:
    """Compute 384-dim embedding for a string."""
    model = _get_model()
    return model.encode(text, convert_to_numpy=True).tolist()


async def get_or_compute_embedding(
    prompt: str,
    supabase_client,
) -> tuple[list[float], str, bool]:
    """
    Returns (embedding_vector, prompt_hash, was_cached).
    Writes to Supabase if cache miss.
    """
    prompt_hash = compute_prompt_hash(prompt)

    # Cache lookup
    result = (
        supabase_client.table("prompt_embeddings")
        .select("embedding")
        .eq("prompt_hash", prompt_hash)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if rows and rows[0].get("embedding"):
        return rows[0]["embedding"], prompt_hash, True

    # Cache miss — compute
    vector = embed_text(prompt)

    # Store (upsert by prompt_hash)
    supabase_client.table("prompt_embeddings").upsert({
        "prompt_hash": prompt_hash,
        "embedding":   vector,
    }, on_conflict="prompt_hash").execute()

    return vector, prompt_hash, False
```

Install dependency:
```bash
pip install sentence-transformers
```

Add to `backend/requirements.txt`:
```
sentence-transformers>=2.7.0
```

---

### P2.3 — Add KNN search function

**New file:** `backend/services/knn_search.py`

```python
"""
KNN similarity search against prompt_embeddings + benchmark_results.
"""
from __future__ import annotations
from statistics import median, stdev
from typing import Optional


MIN_NEIGHBOR_SIMILARITY = 0.72
DEFAULT_K              = 20
FALLBACK_K             = 40
FALLBACK_SIMILARITY    = 0.60
MIN_MODEL_NEIGHBORS    = 3   # min rows per model within KNN results


def search_neighbors(
    supabase_client,
    embedding: list[float],
    use_case: str,
    k: int = DEFAULT_K,
    min_similarity: float = MIN_NEIGHBOR_SIMILARITY,
) -> list[dict]:
    """
    Calls Supabase RPC knn_search(query_embedding, use_case, k, min_sim).
    Returns rows with: model_id, provider, avg_accuracy_score, cost,
                       latency_ms, similarity, eval_conflict_flag, low_confidence
    """
    result = supabase_client.rpc("knn_search", {
        "query_embedding": embedding,
        "target_use_case": use_case,
        "result_limit":    k,
        "min_similarity":  min_similarity,
    }).execute()
    return result.data or []


def aggregate_knn_signals(neighbors: list[dict]) -> dict[str, dict]:
    """
    For each model_id in neighbors, compute:
      - sim_weighted_accuracy
      - p50_cost
      - p50_latency
      - score_variance
      - sample_n (within neighbors)
    Returns { model_id: { signals } }
    """
    grouped: dict[str, list[dict]] = {}
    for row in neighbors:
        mid = row["model_id"]
        grouped.setdefault(mid, []).append(row)

    aggregated = {}
    for model_id, rows in grouped.items():
        if len(rows) < MIN_MODEL_NEIGHBORS:
            continue  # too sparse — caller will inject prior

        sims    = [r["similarity"]         for r in rows]
        accs    = [r["avg_accuracy_score"] for r in rows]
        costs   = [r["cost"]               for r in rows]
        lats    = [r["latency_ms"]         for r in rows]

        total_sim = sum(sims)
        sim_weighted_acc = sum(s * a for s, a in zip(sims, accs)) / total_sim

        aggregated[model_id] = {
            "model_id":            model_id,
            "provider":            rows[0]["provider"],
            "sim_weighted_accuracy": round(sim_weighted_acc, 2),
            "p50_cost":            round(median(costs), 6),
            "p50_latency":         round(median(lats), 1),
            "score_variance":      round(stdev(accs) if len(accs) > 1 else 0, 2),
            "sample_n":            len(rows),
        }

    return aggregated
```

**Supabase RPC function (run in SQL editor):**
```sql
CREATE OR REPLACE FUNCTION knn_search(
    query_embedding  vector(384),
    target_use_case  TEXT,
    result_limit     INT     DEFAULT 20,
    min_similarity   FLOAT   DEFAULT 0.72
)
RETURNS TABLE (
    row_id              UUID,
    model_id            TEXT,
    provider            TEXT,
    avg_accuracy_score  FLOAT,
    cost                FLOAT,
    latency_ms          FLOAT,
    similarity          FLOAT,
    eval_conflict_flag  BOOLEAN,
    low_confidence      BOOLEAN
)
LANGUAGE SQL STABLE AS $$
    SELECT
        br.id                AS row_id,
        br.model_id,
        br.provider,
        br.avg_accuracy_score,
        br.cost,
        br.latency_ms,
        1 - (pe.embedding <=> query_embedding) AS similarity,
        COALESCE(br.eval_conflict_flag, FALSE)  AS eval_conflict_flag,
        COALESCE(br.low_confidence, FALSE)      AS low_confidence
    FROM prompt_embeddings pe
    JOIN benchmark_results br ON pe.prompt_hash = br.prompt_hash
    WHERE br.use_case = target_use_case
      AND br.avg_accuracy_score IS NOT NULL
      AND COALESCE(br.low_confidence, FALSE) = FALSE
      AND 1 - (pe.embedding <=> query_embedding) >= min_similarity
    ORDER BY pe.embedding <=> query_embedding
    LIMIT result_limit;
$$;
```

---

### P2.4 — Shadow mode: run KNN alongside current recommender

Modify `recommender.py` to run KNN in the background and log the comparison.
Do NOT change what the user sees yet.

```python
async def get_recommendation(use_case: str, prompt: str, current_model: str) -> dict:
    # ... existing logic (unchanged) ...
    result = { ... existing output ... }

    # ★ SHADOW MODE: run KNN in background for comparison logging
    asyncio.create_task(
        _shadow_knn_recommendation(prompt, use_case, result)
    )

    return result


async def _shadow_knn_recommendation(prompt: str, use_case: str, slice_result: dict):
    """
    Runs KNN recommendation silently. Logs agreement/disagreement with slice.
    Does NOT affect the response returned to the user.
    """
    try:
        from services.embedding_service import get_or_compute_embedding
        from services.knn_search import search_neighbors, aggregate_knn_signals
        from services.supabase_client import supabase

        vector, prompt_hash, was_cached = await get_or_compute_embedding(prompt, supabase)
        neighbors = search_neighbors(supabase, vector, use_case)

        if not neighbors:
            print(f"[SHADOW KNN] No neighbors found for use_case={use_case}")
            return

        signals   = aggregate_knn_signals(neighbors)
        knn_top   = max(signals.values(), key=lambda x: x["sim_weighted_accuracy"])
        slice_top = slice_result.get("recommended_model", "?")

        agree = knn_top["model_id"] == slice_top
        print(
            f"[SHADOW KNN] knn={knn_top['model_id']} (acc={knn_top['sim_weighted_accuracy']}) "
            f"slice={slice_top} agree={agree} neighbors={len(neighbors)}"
        )
    except Exception as exc:
        print(f"[SHADOW KNN ERROR] {exc}")
```

**Monitoring:** Watch the logs for 1–2 weeks. You want to see:
- Agreement rate ≥ 75% → KNN is consistent enough to promote
- KNN has neighbors for prompts where `filter_level = 'use_case_only'` → proves KNN helps sparse cases
- No latency spikes (KNN should add < 50ms to requests via HNSW index)

---

### P2.5 — Batch-compute embeddings for all existing rows

```python
# model_training/backfill_embeddings.py
"""
One-time script to compute and store embeddings for all benchmark rows
that have prompt_hash but no embedding in prompt_embeddings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv(Path(__file__).parent.parent / "backend" / ".env")

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
model    = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Get all unique prompt_hashes that need embedding
existing = {row["prompt_hash"] for row in
            supabase.table("prompt_embeddings").select("prompt_hash").execute().data or []}

start = 0
PAGE  = 500
while True:
    rows = (supabase.table("benchmark_results")
            .select("prompt_hash,prompt")
            .not_.is_("prompt_hash", "null")
            .range(start, start + PAGE - 1)
            .execute().data or [])
    if not rows:
        break

    to_embed = [r for r in rows if r["prompt_hash"] not in existing]
    if to_embed:
        texts   = [r["prompt"] for r in to_embed]
        vectors = model.encode(texts, batch_size=64, show_progress_bar=True).tolist()
        for row, vec in zip(to_embed, vectors):
            supabase.table("prompt_embeddings").upsert({
                "prompt_hash": row["prompt_hash"],
                "embedding":   vec,
            }, on_conflict="prompt_hash").execute()
            existing.add(row["prompt_hash"])

    start += PAGE
    print(f"Processed: {start}")

print("Embedding backfill complete.")
```

---

## PHASE 3 — Promote KNN to Primary
### Timeline: 3–5 days | Risk: Medium | Prerequisite: P2 shadow logs OK

---

### P3.1 — Create model_priors table

```sql
CREATE TABLE IF NOT EXISTS model_priors (
    model_id         TEXT,
    use_case         TEXT,
    prompt_complexity TEXT,
    prior_accuracy   FLOAT,
    prior_cost       FLOAT,
    prior_latency    FLOAT,
    support_n        INTEGER,
    last_updated     TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (model_id, use_case, prompt_complexity)
);
```

**Populate it (run weekly via scheduled job):**
```sql
INSERT INTO model_priors
    (model_id, use_case, prompt_complexity, prior_accuracy,
     prior_cost, prior_latency, support_n, last_updated)
SELECT
    model_id,
    use_case,
    prompt_complexity,
    AVG(avg_accuracy_score)   AS prior_accuracy,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cost)       AS prior_cost,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms) AS prior_latency,
    COUNT(*)                  AS support_n,
    now()
FROM benchmark_results
WHERE avg_accuracy_score IS NOT NULL
  AND low_confidence = FALSE
GROUP BY model_id, use_case, prompt_complexity
ON CONFLICT (model_id, use_case, prompt_complexity)
DO UPDATE SET
    prior_accuracy = EXCLUDED.prior_accuracy,
    prior_cost     = EXCLUDED.prior_cost,
    prior_latency  = EXCLUDED.prior_latency,
    support_n      = EXCLUDED.support_n,
    last_updated   = now();
```

---

### P3.2 — Create routing_log table

```sql
CREATE TABLE IF NOT EXISTS routing_log (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id        TEXT,
    prompt_hash       TEXT,
    use_case          TEXT,
    complexity        TEXT,
    clarity           TEXT,
    recommended_model TEXT,
    data_source       TEXT,   -- 'knn' | 'prior' | 'slice'
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

### P3.3 — Replace slice-based recommender with KNN-primary

Replace the core of `get_recommendation()` in `recommender.py`:

```
CURRENT FLOW:
  complexity → clarity → load all rows → slice → summarize → rank → switch

TARGET FLOW:
  complexity → clarity → [embed in parallel] → KNN → aggregate →
  inject priors for sparse models → rule engine → score fusion → switch
```

Full rewrite of `get_recommendation()`:

```python
async def get_recommendation(use_case: str, prompt: str, current_model: str) -> dict:
    import uuid
    from services.embedding_service import get_or_compute_embedding
    from services.knn_search import search_neighbors, aggregate_knn_signals
    from services.supabase_client import supabase

    request_id = str(uuid.uuid4())[:8]

    # Step 1+2 in parallel
    classifier = load_complexity_classifier()
    complexity_task = asyncio.create_task(
        asyncio.coroutine(lambda: infer_complexity(prompt, use_case, classifier))()
    )
    embedding_task = asyncio.create_task(
        get_or_compute_embedding(prompt, supabase)
    )
    (complexity, complexity_confidence, complexity_source), \
    (vector, prompt_hash, was_cached) = await asyncio.gather(
        asyncio.coroutine(lambda: infer_complexity(prompt, use_case, classifier))(),
        get_or_compute_embedding(prompt, supabase),
    )
    clarity, clarity_source = await infer_clarity(prompt, use_case)

    # Step 3 — KNN
    neighbors = search_neighbors(supabase, vector, use_case)

    # Step 3b — OOD fallback: too few neighbors
    if len(neighbors) < 5:
        neighbors = search_neighbors(
            supabase, vector, use_case,
            k=40, min_similarity=0.60
        )

    data_source = "knn" if len(neighbors) >= 5 else "prior"

    # Step 4 — Aggregate KNN signals
    knn_signals = aggregate_knn_signals(neighbors)   # { model_id: signals }

    # Step 5 — Inject priors for models with < 3 KNN neighbors
    all_model_ids = get_model_ids_for_use_case(use_case)
    priors = _load_priors(supabase, use_case, complexity, list(all_model_ids))

    candidates = {}
    for model_id in all_model_ids:
        if model_id in knn_signals:
            candidates[model_id] = knn_signals[model_id]
        elif model_id in priors:
            candidates[model_id] = {**priors[model_id], "from_prior": True}
        # else: not enough data at all, skip

    if not candidates:
        raise ValueError("No candidate models found for this use case.")

    # Step 6 — Score fusion
    ranked = _score_and_rank(candidates)
    best   = ranked[0]

    # Step 7 — Switching policy
    current_stats     = candidates.get(current_model)
    switch, policy_reason = should_switch_v2(best, current_stats)
    final_model = best["model_id"] if switch else current_model

    # Step 8 — Async log
    asyncio.create_task(_write_routing_log(supabase, {
        "request_id":        request_id,
        "prompt_hash":       prompt_hash,
        "use_case":          use_case,
        "complexity":        complexity,
        "clarity":           clarity,
        "recommended_model": best["model_id"],
        "data_source":       data_source,
        "knn_neighbors":     len(neighbors),
        "expected_accuracy": best.get("sim_weighted_accuracy") or best.get("prior_accuracy"),
    }))

    return {
        "complexity":         complexity,
        "complexity_source":  complexity_source,
        "clarity":            clarity,
        "clarity_source":     clarity_source,
        "data_source":        data_source,
        "knn_neighbors_used": len(neighbors),
        "recommended_model":  best["model_id"],
        "recommended_provider": best["provider"],
        "expected_accuracy":  best.get("sim_weighted_accuracy") or best.get("prior_accuracy"),
        "expected_cost":      best.get("p50_cost") or best.get("prior_cost"),
        "expected_latency":   best.get("p50_latency") or best.get("prior_latency"),
        "switch_recommended": switch,
        "policy_reason":      policy_reason,
        "final_suggestion_model": final_model,
        "top_candidates":     ranked[:5],
    }
```

---

### P3 Validation

```sql
-- Routing log has data
SELECT data_source, COUNT(*) FROM routing_log
GROUP BY data_source;
-- Expect: 'knn' >> 'prior' (priors should be minority fallback)

-- KNN agreement check (compare to shadow logs)
-- If you ran shadow mode for 2 weeks, compute agreement:
SELECT
    SUM(CASE WHEN knn_data_source = slice_data_source THEN 1 ELSE 0 END)::float
    / COUNT(*) AS agreement_rate
FROM shadow_comparison_log;
-- Target: >= 0.75
```

---

## Summary Table

| Phase | Duration | What Changes | User Impact |
|---|---|---|---|
| **P0** Score fix | 1–2 days | `recommender.py`, `generate_avg_accuracy_scores.py` | Immediately better routing quality |
| **P1** Quality flags | 3–5 days | Schema columns, evaluation pipeline | Noisy rows excluded from routing |
| **P2** KNN shadow | 1–2 weeks | Embedding service, KNN search, shadow logging | Zero (runs in background only) |
| **P3** KNN primary | 3–5 days | Recommender core rewrite, priors table, routing log | Semantic-aware recommendations |

---

## Key Numbers to Track

| Metric | Current | P0 Target | P3 Target |
|---|---|---|---|
| % rows with `avg_accuracy_score` | ~40% | 100% | 100% |
| % rows with `low_confidence=FALSE` | 0% (not tracked) | 80%+ | 85%+ |
| Recommendation latency (P95) | ~800ms (Supabase fetch) | ~800ms | <250ms (KNN index) |
| Cold-start rate (filter_level=use_case_only) | Unknown | <20% | <5% (priors cover it) |
| KNN vs slice agreement | N/A | N/A (shadow) | ≥75% |

# LLM Router — Architecture & Implementation Plan
### Response to Manager Audit + Accuracy-by-LLM Audit

---

> **Core diagnosis from both audits:**
> `avg_accuracy_score` is near-zero-variance noise (everything clusters 91–97).
> The 55% accuracy weight in score fusion is contributing almost nothing to model differentiation.
> The real discriminators are cost and latency — but they're only weighted 25% and 15%.
> **Fix: replace absolute scoring with pairwise win rate as the primary signal.**

---

## PART 1 — THE PROBLEM IN ONE DIAGRAM

```
CURRENT STATE (broken signal path)
────────────────────────────────────

Prompt → KNN → 39 neighbors → per-model accuracy (e.g. 99.38 vs 98.23)
                                       ↑
                             LLM judge scores 0-100
                             (inflated, clustered 91-97)
                             (std ≈ 3 points across 100-point scale)
                             (55% weight → near-zero discrimination)
                                       ↓
              score = 0.55×acc_norm + 0.25×cost_norm + 0.15×lat_norm
                      ≈ 0.55×NOISE  + real signal
              → recommendation driven mostly by cost/latency despite weights


TARGET STATE (real signal path)
────────────────────────────────────

Prompt → KNN → 39 neighbors → per-model win_rate (e.g. 0.70 vs 0.42)
                                       ↑
                             Pairwise judge: "A vs B, which is better?"
                             (binary, harder to inflate)
                             (real discrimination: 70% vs 42% is meaningful)
                             + code_syntax_pass_rate (code tasks only)
                             + consistency_score (all tasks)
                                       ↓
          score = 0.55×win_rate_norm + 0.25×cost_norm + 0.15×lat_norm + 0.05×conf
                         ↑ now meaningful discrimination
```

---

## PART 2 — NEW DATABASE SCHEMA

### 2.1 `pairwise_results` table (NEW — primary signal source)

```sql
CREATE TABLE IF NOT EXISTS pairwise_results (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_hash     TEXT        NOT NULL,
    use_case        TEXT        NOT NULL,
    complexity      TEXT,                    -- low / mid / high
    model_a         TEXT        NOT NULL,
    model_b         TEXT        NOT NULL,
    response_a      TEXT,
    response_b      TEXT,
    winner          TEXT        NOT NULL,    -- 'A' | 'B' | 'TIE'
    winner_model    TEXT        NOT NULL,    -- short_id of winning model
    loser_model     TEXT        NOT NULL,
    judge_model     TEXT        NOT NULL,    -- short_id of evaluator
    reason          TEXT,                    -- one-sentence from judge
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON pairwise_results (prompt_hash);
CREATE INDEX ON pairwise_results (use_case, complexity);
CREATE INDEX ON pairwise_results (model_a, model_b);
CREATE INDEX ON pairwise_results (winner_model);
CREATE INDEX ON pairwise_results (created_at DESC);
```

### 2.2 `model_win_rates` table (NEW — materialized view, refreshed nightly)

```sql
CREATE TABLE IF NOT EXISTS model_win_rates (
    model_id        TEXT        NOT NULL,
    use_case        TEXT        NOT NULL,
    complexity      TEXT        NOT NULL,   -- low / mid / high / all
    win_rate        FLOAT       NOT NULL,   -- 0.0–1.0
    total_matches   INTEGER     NOT NULL,
    wins            INTEGER     NOT NULL,
    losses          INTEGER     NOT NULL,
    ties            INTEGER     NOT NULL,
    judge_count     INTEGER,               -- distinct judges used
    last_updated    TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (model_id, use_case, complexity)
);

CREATE INDEX ON model_win_rates (use_case, complexity, win_rate DESC);
```

### 2.3 Additions to `benchmark_results` (existing table)

```sql
-- Run once in Supabase SQL Editor
ALTER TABLE benchmark_results
    ADD COLUMN IF NOT EXISTS syntax_pass       BOOLEAN,    -- code tasks only
    ADD COLUMN IF NOT EXISTS syntax_checked    BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS consistency_score FLOAT,      -- 0.0–1.0, optional
    ADD COLUMN IF NOT EXISTS win_rate          FLOAT;      -- denormalized from model_win_rates

-- Index for fast KNN signal aggregation
CREATE INDEX ON benchmark_results (model_id, use_case, syntax_pass)
    WHERE avg_accuracy_score IS NOT NULL;
```

### 2.4 `routing_log` table (must be created — BUG-001)

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

## PART 3 — NEW METRICS PIPELINE

### 3.1 Pairwise Evaluation Script

**File:** `model_training/run_pairwise_eval.py`

```
Purpose:
  For each unique prompt_hash in benchmark_results,
  take top-5 models by avg_accuracy_score,
  run all C(5,2) = 10 pairwise comparisons,
  store winner/loser in pairwise_results.

Judge model: llama4-maverick (primary), mistral-large (secondary)

Pairwise prompt template:
─────────────────────────────────────────────────────────────
System: You are evaluating two LLM responses. Be critical and decisive.

Prompt given to both models: {prompt}

Response A ({model_a}): {response_a}
Response B ({model_b}): {response_b}

Which response better addresses the prompt? Consider:
accuracy, completeness, and conciseness.

Output ONLY valid JSON: {"winner": "A" or "B", "reason": "one sentence"}
─────────────────────────────────────────────────────────────

Anti-bias measures:
  - Randomize which model is A vs B
  - Use 2 judges per pair, award tie if they disagree
  - Store which model was A/B for audit trail
```

**Architecture:**

```
benchmark_results (548 prompts × 14 models = stored responses)
        │
        ▼
run_pairwise_eval.py
    ├── Fetch all (prompt_hash, use_case, model_id, response) groups
    ├── For each prompt: take top-5 models
    ├── Generate C(5,2) = 10 pairs
    ├── For each pair: call pairwise judge × 2 (parallel)
    │     ├── Judge 1: llama4-maverick → winner A/B
    │     └── Judge 2: mistral-large  → winner A/B
    │           ├── Both agree → record winner
    │           └── Disagree  → record "TIE"
    └── Upsert to pairwise_results
```

**Estimated cost:**
- 548 prompts × 10 pairs × 2 judges = 10,960 evaluation calls
- Each call ~500 tokens → ~5.5M tokens total
- At llama4-maverick pricing: ~$4.94 total

### 3.2 Win Rate Computation Script

**File:** `model_training/compute_win_rates.py`

```sql
-- SQL that this script runs to refresh model_win_rates:

INSERT INTO model_win_rates
    (model_id, use_case, complexity, win_rate, total_matches, wins, losses, ties, judge_count, last_updated)

SELECT
    winner_model                        AS model_id,
    use_case,
    COALESCE(complexity, 'all')         AS complexity,
    COUNT(CASE WHEN winner != 'TIE' THEN 1 END)::float
        / NULLIF(COUNT(*), 0)           AS win_rate,
    COUNT(*)                            AS total_matches,
    COUNT(CASE WHEN winner_model = model_a OR winner_model = model_b 
               THEN 1 END)             AS wins,
    COUNT(CASE WHEN winner = 'TIE' THEN 1
               WHEN winner_model != winner_model THEN 1
               END)                    AS losses,
    COUNT(CASE WHEN winner = 'TIE' THEN 1 END) AS ties,
    COUNT(DISTINCT judge_model)         AS judge_count
FROM (
    SELECT prompt_hash, use_case, complexity,
           winner_model, loser_model, judge_model,
           model_a, model_b, winner
    FROM pairwise_results
    UNION ALL
    SELECT prompt_hash, use_case, complexity,
           loser_model AS winner_model, winner_model AS loser_model,
           judge_model, model_a, model_b, winner
    FROM pairwise_results
) both_sides
GROUP BY winner_model, use_case, COALESCE(complexity, 'all')
ON CONFLICT (model_id, use_case, complexity)
DO UPDATE SET
    win_rate      = EXCLUDED.win_rate,
    total_matches = EXCLUDED.total_matches,
    wins          = EXCLUDED.wins,
    losses        = EXCLUDED.losses,
    ties          = EXCLUDED.ties,
    judge_count   = EXCLUDED.judge_count,
    last_updated  = now();
```

### 3.3 Code Syntax Verification

**File:** `model_training/verify_syntax.py`

```python
import ast
import re
from supabase import create_client

def check_code_syntax(response: str) -> bool | None:
    """
    Returns True  → valid Python syntax found in code block
    Returns False → syntax error found
    Returns None  → no code block (not applicable)
    """
    blocks = re.findall(r'```(?:python)?\n(.*?)```', response, re.DOTALL)
    if not blocks:
        return None
    try:
        ast.parse(blocks[0])
        return True
    except SyntaxError:
        return False

# Run against all code-generation rows where syntax_checked = False
# Updates benchmark_results.syntax_pass + syntax_checked = True
```

---

## PART 4 — DIVERSE PROMPT DATASET GENERATION

### 4.1 The Problem

Current corpus: **548 unique prompts** backing 8,594 rows.
- Average 15.7 rows per prompt (same prompt, different models)
- KNN retrieves identical prompts from different angles → clustering artifact
- Out-of-distribution prompts (legal, medical, multilingual) → no neighbors

### 4.2 Target Corpus

| Dimension | Current | Target |
|---|---|---|
| Unique prompts | 548 | **2,500** |
| Use cases | 3 | **5** |
| Complexities per use case | 3 | **3** |
| Domains per use case | ~2 | **8–10** |
| Languages | EN only | EN + code comments in other langs |

### 4.3 Use Case × Complexity × Domain Matrix

**File:** `model_training/prompt_generator/prompt_matrix.py`

```
USE CASE: code-generation (target: 600 unique prompts)
───────────────────────────────────────────────────────
  low:    50 prompts × 6 domains = 300 total
    domains: algorithms, string manipulation, math utils,
             file I/O, API calls, data structures

  mid:    30 prompts × 6 domains = 180 total
    domains: web scraping, ORM queries, async patterns,
             REST API design, test writing, CLI tools

  high:   20 prompts × 6 domains = 120 total
    domains: distributed systems, concurrency, LLM tool calling,
             system design implementation, optimization problems

USE CASE: reasoning (target: 400 unique prompts)
───────────────────────────────────────────────────────
  low:    50 prompts × 4 domains = 200
    domains: math word problems, logic puzzles, basic inference,
             fact checking

  mid:    30 prompts × 3 domains = 90
    domains: causal reasoning, multi-step problem solving,
             argument analysis

  high:   20 prompts × 3 domains = 60
    domains: formal logic, Bayesian reasoning, complex analogies,
             research-level problem decomposition

  verifiable: 50 prompts with numeric/checkable answers
    → These get binary correct/incorrect signal, no judge needed

USE CASE: text-generation (target: 500 unique prompts)
───────────────────────────────────────────────────────
  low:    80 prompts × 4 types = 320
    types: short email, 1-para summary, simple explanation,
           quick definition

  mid:    30 prompts × 3 types = 90
    types: technical article, product description, report section

  high:   15 prompts × 6 types = 90
    types: persuasive essay, white paper section, press release,
           executive summary, legal brief summary, grant proposal

USE CASE: data-analysis (NEW — target: 400 prompts)
───────────────────────────────────────────────────────
  low:    pandas operations, simple SQL queries, CSV parsing
  mid:    aggregation queries, data cleaning pipelines,
          chart recommendation, schema design
  high:   time-series analysis, statistical modeling,
          dashboard architecture, optimization queries

USE CASE: question-answering (NEW — target: 600 prompts)
───────────────────────────────────────────────────────
  low:    factual questions with verifiable answers
  mid:    explanatory questions, comparison questions
  high:   research synthesis, multi-document QA
```

### 4.4 Prompt Generation Pipeline

**File:** `model_training/prompt_generator/generate_prompts.py`

```
Step 1: Seed generation (LLM-assisted)
  → Use llama4-maverick to generate 50 candidate prompts per cell
  → Deduplicate by embedding similarity (cosine > 0.85 → keep only one)
  → Filter: too short (<10 words), too long (>150 words), duplicates

Step 2: Diversity check
  → Embed all generated prompts (OpenAI text-embedding-3-small)
  → Cluster at k=50 per use case
  → Ensure each cluster has at least 5 prompts
  → If a cluster has <3 prompts → generate more

Step 3: Verifiable answer tagging (for reasoning + QA)
  → For math/logic prompts: generate reference answer
  → Store in benchmark_results.reference_answer column
  → Use binary correct/incorrect instead of LLM score

Step 4: Output
  → Write to model_training/prompts_v2.csv with columns:
     prompt, use_case, complexity, domain, has_verifiable_answer,
     reference_answer, prompt_hash
```

### 4.5 New Columns for `benchmark_results`

```sql
ALTER TABLE benchmark_results
    ADD COLUMN IF NOT EXISTS domain              TEXT,     -- subdomain label
    ADD COLUMN IF NOT EXISTS has_ref_answer      BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS reference_answer    TEXT,
    ADD COLUMN IF NOT EXISTS is_correct          BOOLEAN,  -- for verifiable prompts
    ADD COLUMN IF NOT EXISTS prompt_version      INTEGER DEFAULT 1;  -- v1=old, v2=new
```

---

## PART 5 — RECOMMENDER CHANGES

### 5.1 New Score Fusion Formula (per use_case)

**File:** `backend/services/recommender.py` → `score_and_rank_knn_candidates()`

```python
# Current (broken — accuracy is noise):
score = 0.55 * acc_norm + 0.25 * cost_norm + 0.15 * lat_norm + 0.05 * conf

# New — per use_case weights:
SCORE_WEIGHTS = {
    "code-generation": {
        "win_rate":    0.40,   # pairwise win rate (primary)
        "syntax_rate": 0.20,   # code-specific: % syntax-valid responses
        "cost":        0.25,   # still matters
        "latency":     0.10,   # less important for code (async tasks)
        "confidence":  0.05,
    },
    "reasoning": {
        "win_rate":    0.50,   # pairwise win rate
        "correctness": 0.20,   # binary correct/incorrect where verifiable
        "cost":        0.20,
        "latency":     0.05,
        "confidence":  0.05,
    },
    "text-generation": {
        "win_rate":    0.45,   # pairwise win rate
        "cost":        0.25,
        "latency":     0.20,
        "confidence":  0.10,
    },
    "data-analysis": {
        "win_rate":    0.45,
        "cost":        0.30,
        "latency":     0.20,
        "confidence":  0.05,
    },
    "question-answering": {
        "win_rate":    0.45,
        "correctness": 0.20,
        "cost":        0.25,
        "latency":     0.05,
        "confidence":  0.05,
    },
    # Fallback for unknown use cases:
    "_default": {
        "win_rate":    0.45,
        "cost":        0.30,
        "latency":     0.20,
        "confidence":  0.05,
    },
}
```

### 5.2 New KNN Signal Aggregation

**Current** (`aggregate_knn_signals`):
```python
# Groups neighbors by model, computes sim_weighted_accuracy
# Problem: accuracy has near-zero variance → weights pure noise
```

**New** (`aggregate_knn_signals_v2`):
```python
def aggregate_knn_signals_v2(neighbors: list[dict], use_case: str) -> list[dict]:
    """
    For each model in the KNN neighbor set, aggregate:
      - win_rate: pulled from model_win_rates table (not KNN-dependent)
      - syntax_pass_rate: % of its code-gen rows with syntax_pass=True
      - correctness_rate: % correct on verifiable reasoning prompts
      - p50_cost, p50_latency: from KNN rows
      - sample_n, avg_similarity: for confidence

    Signal priority by use_case (see SCORE_WEIGHTS above).
    Falls back to avg_accuracy_score if win_rate not yet available.
    """
    ...
```

### 5.3 Win Rate Join (new Supabase query)

**File:** `backend/services/supabase_client.py` — new function:

```python
async def get_model_win_rates(
    use_case: str,
    complexity: Optional[str] = None,
    min_matches: int = 10,
) -> dict[str, float]:
    """
    Returns {model_id: win_rate} for the given use_case+complexity slice.
    Only returns models with at least min_matches pairwise comparisons.
    Falls back to 'all' complexity if specific slice has insufficient data.
    """
    rows = (
        supabase.table("model_win_rates")
        .select("model_id, win_rate, total_matches")
        .eq("use_case", use_case)
        .eq("complexity", complexity or "all")
        .gte("total_matches", min_matches)
        .execute()
    ).data or []

    if not rows and complexity:
        # Fallback to 'all' complexity
        rows = (
            supabase.table("model_win_rates")
            .select("model_id, win_rate, total_matches")
            .eq("use_case", use_case)
            .eq("complexity", "all")
            .gte("total_matches", min_matches)
            .execute()
        ).data or []

    return {r["model_id"]: r["win_rate"] for r in rows}
```

### 5.4 Switching Policy Overhaul

**Current (broken):** `min_accuracy_gain = 2.0` — threshold on inflated 91-97 scores, never meaningfully triggered.

**New:**
```python
# Switch only if win-rate advantage is material
MIN_WIN_RATE_ADVANTAGE = 0.10    # recommended model must win 10pp more often
MIN_COST_IMPROVEMENT_PCT = 15.0  # no change here
MIN_LATENCY_IMPROVEMENT_PCT = 20.0

def should_switch(
    recommended: dict,
    current: dict,
    mode: str = "best_value",
) -> tuple[bool, str]:
    """
    Returns (switch, reason).
    Uses win_rate advantage as primary gate instead of accuracy delta.
    """
    win_rate_delta = recommended.get("win_rate", 0) - current.get("win_rate", 0)
    cost_delta_pct = pct_change(current["cost"], recommended["cost"])
    lat_delta_pct  = pct_change(current["latency"], recommended["latency"])

    # Must win more on quality OR be meaningfully cheaper/faster
    quality_win = win_rate_delta >= MIN_WIN_RATE_ADVANTAGE
    cost_win    = cost_delta_pct <= -MIN_COST_IMPROVEMENT_PCT   # negative = cheaper
    speed_win   = lat_delta_pct  <= -MIN_LATENCY_IMPROVEMENT_PCT

    if quality_win:
        return True, f"Win rate advantage: +{win_rate_delta:.0%}"
    if cost_win and win_rate_delta >= -0.05:  # not meaningfully worse on quality
        return True, f"Cost savings: {cost_delta_pct:.0%} with comparable quality"
    if speed_win and win_rate_delta >= -0.05:
        return True, f"Latency improvement: {lat_delta_pct:.0%} with comparable quality"

    return False, "Recommended model not materially better than current"
```

---

## PART 6 — IMPLEMENTATION SEQUENCE

### Phase 0 — Unblock Observability (1 hour)

**Action:** Run SQL in Supabase SQL Editor

```
1. Create routing_log table (see Part 2.4 SQL)
2. Create pairwise_results table (see Part 2.1 SQL)
3. Create model_win_rates table (see Part 2.2 SQL)
4. ALTER TABLE benchmark_results ADD columns (see Part 2.3 + 4.5 SQL)
→ After this: telemetry starts flowing. All future recs get logged.
```

### Phase 1 — Pairwise Evaluation on Existing Data (Day 1)

**Morning (3h):**
```
File: model_training/run_pairwise_eval.py

1. Fetch all 548 unique prompt_hashes from benchmark_results
2. For each hash: fetch all (model_id, response) pairs
3. Take top-5 models by avg_accuracy_score
4. Generate C(5,2)=10 pairs with randomized A/B assignment
5. Call pairwise judge (llama4-maverick) × 2 for each pair
6. Upsert results to pairwise_results table
   Estimated time: ~4h with concurrency=10
   Estimated cost: ~$5
```

**Afternoon (1.5h):**
```
File: model_training/compute_win_rates.py

1. Run SQL from Part 3.2 to populate model_win_rates
2. Verify: SELECT model_id, use_case, win_rate, total_matches
           FROM model_win_rates ORDER BY win_rate DESC;
3. Should see meaningful spread: e.g. llama4-maverick 0.68, nova-pro 0.52
```

### Phase 2 — Syntax Verification (Day 1 Afternoon)

```
File: model_training/verify_syntax.py

1. Fetch all code-generation rows where syntax_checked = FALSE
2. Run check_code_syntax() on each response
3. UPDATE benchmark_results SET syntax_pass=X, syntax_checked=TRUE
   → Expect ~85-95% pass rate for production models
   → Models with <80% syntax pass rate flagged as unreliable for code tasks
```

### Phase 3 — Wire Win Rate into Recommender (Day 2)

**Morning (3h):**
```
File: backend/services/recommender.py

Changes:
1. Add get_model_win_rates() call at start of get_recommendation()
2. Pass win_rates dict to aggregate_knn_signals_v2()
3. Replace score fusion formula per SCORE_WEIGHTS (Part 5.1)
4. Replace switching policy per Part 5.4
5. Add win_rate field to API response

File: backend/services/supabase_client.py
Changes:
1. Add get_model_win_rates() function (Part 5.3)
```

**Afternoon (2h):**
```
File: backend/models/schemas.py
Changes:
1. Add win_rate, syntax_pass_rate fields to InferenceResponse
2. Update response builder to include new signals

Test:
  curl -X POST http://localhost:8000/api/inference/recommend \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Write a binary search in Python","use_case":"code-generation"}'

  Expected: recommended model has highest win_rate for code-gen
```

### Phase 4 — Diverse Prompt Generation (Day 3–4)

**Day 3 (3h):**
```
File: model_training/prompt_generator/generate_prompts.py

1. Define domain matrix (Part 4.3)
2. Call llama4-maverick to generate 50 candidates per cell
3. Embed with OpenAI text-embedding-3-small
4. Deduplicate: if cosine_sim > 0.85 with existing prompt → discard
5. Target: ~2,500 unique prompts across 5 use cases
6. Write to model_training/prompts_v2.csv
```

**Day 4 (ongoing pipeline):**
```
File: model_training/run_benchmarks_v2.py

1. Read prompts_v2.csv
2. Call all 14 Bedrock models per prompt (existing infrastructure)
3. Store responses in benchmark_results with prompt_version=2
4. Trigger pairwise evaluation immediately after each batch
5. Auto-compute win rates nightly
→ After 1 week: 2,500 × 14 = 35,000 new rows with pairwise data
```

### Phase 5 — Recommender Validation (After Phase 3–4)

```
Run ab_test.py after new data is in:
  python model_training/ab_test.py --input prompts_v2_sample.csv --concurrency 5

Key metric: Group B (KNN) should have avg win_rate > Group A (control)
If B.win_rate > A.win_rate by >10pp → system is working
```

---

## PART 7 — EXACT FILE CHANGES

| File | Change | Priority |
|---|---|---|
| Supabase SQL Editor | Create 4 new tables/columns | 🔴 Do first |
| `model_training/run_pairwise_eval.py` | New file — run pairwise judge | 🔴 Day 1 |
| `model_training/compute_win_rates.py` | New file — refresh win_rates table | 🔴 Day 1 |
| `model_training/verify_syntax.py` | New file — code syntax check | 🟠 Day 1 |
| `backend/services/supabase_client.py` | Add `get_model_win_rates()` | 🔴 Day 2 |
| `backend/services/recommender.py` | Replace fusion formula + switch policy | 🔴 Day 2 |
| `backend/models/schemas.py` | Add win_rate to response schema | 🟠 Day 2 |
| `model_training/prompt_generator/generate_prompts.py` | New file — diverse prompts | 🟡 Day 3 |
| `model_training/run_benchmarks_v2.py` | New file — benchmark new prompts | 🟡 Day 4 |

---

## PART 8 — SUCCESS METRICS

After Phase 3 (win rate wired in):

| Metric | Before | Target |
|---|---|---|
| Signal variance (accuracy std) | ~3 pts / 100-pt scale | win_rate std ≈ 0.15 (real) |
| Discrimination between models | Essentially zero | 0.40–0.70 win rate spread |
| Switch policy trigger rate | ~100% (always switches) | 30–50% (meaningful switches) |
| Models with <80% code syntax pass | Unknown | Identified + penalized |

After Phase 4 (new prompts):

| Metric | Before | Target |
|---|---|---|
| Unique prompts | 548 | 2,500+ |
| KNN avg neighbors | 39 (same prompts) | 20–40 (truly diverse) |
| Out-of-distribution coverage | 2 use cases | 5 use cases |
| KNN confidence on new prompts | Degrades to slice | ≥ 0.70 |

---

## PART 9 — WHAT NOT TO DO (from both audits)

> [!CAUTION]
> **Don't do these things in the current sprint:**

1. **Don't retrain the classifier** — it's a symptom, not the cause. Fix the data quality first.
2. **Don't implement Bradley-Terry Elo** — needs 10K+ pairwise matches. You have ~10K after Phase 1.
3. **Don't add more evaluator models** — 2 judges per pair is enough if the format is binary.
4. **Don't tune the 0.55/0.25/0.15 weights empirically** — the signal (avg_accuracy) is broken. Tuning weights on broken signal amplifies the problem.
5. **Don't lower MIN_MODEL_NEIGHBORS below 3** — this treats 2-sample estimates as reliable.

> [!NOTE]
> **The minimum viable improvement is:**
> `pairwise_results table exists` + `model_win_rates populated` + `recommender uses win_rate` = trustworthy signal
> Everything else is optimization on top of a working foundation.

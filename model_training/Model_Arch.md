**Overview**

The recommendation system we built is a deterministic benchmark-ranking pipeline. It does not use an LLM to choose the model. Instead, it uses structured benchmark data plus a lightweight complexity classifier and a clarity inference layer to find the most comparable historical slice, summarize candidate model performance, and apply a switching policy.

At a high level, the system answers:

1. What kind of prompt is this?
2. Which benchmark rows are actually comparable?
3. Among sufficiently supported models, which ones are near the top in quality?
4. Among those near-top models, which one gives the best value?
5. Is that improvement large enough to justify switching from the current model?

---

**Architecture Layers**

**1. Input Layer**

User inputs:
- `prompt`
- `use_case`
- `current_model`

These come from the frontend recommendation screen and are sent to:
- `POST /api/inference/recommend`

The use case is user-selected and trusted directly. We do not infer use case in the backend.

Files:
- [Inference.jsx](c:\Users\Musharraf\Documents\POC\frontend\src\pages\Inference.jsx)
- [inference.js](c:\Users\Musharraf\Documents\POC\frontend\src\api\inference.js)
- [inference.py](c:\Users\Musharraf\Documents\POC\backend\routers\inference.py)

---

**2. Signal Inference Layer**

This layer derives the signals needed to find the right benchmark slice.

It computes:
- `prompt_complexity`
- `clarity`

It does not use prompt quality in ranking.

### 2.1 Complexity Inference

Complexity is inferred using a trained local classifier:
- artifact: [classifier.pkl](c:\Users\Musharraf\Documents\POC\model_training\artifacts\classifier.pkl)

How it works:
- The backend loads the pickle from disk.
- It predicts one of:
  - `low`
  - `mid`
  - `high`
- If prediction probabilities are available, it also returns confidence.
- If the classifier is missing or fails, it falls back to heuristics.

Heuristic fallback:
- very short prompts with no complexity markers -> `low`
- prompts with architecture / distributed / benchmark / production-like language -> `high`
- otherwise -> `mid`

Files:
- [recommender.py](c:\Users\Musharraf\Documents\POC\backend\services\recommender.py)
- [recommend_v2.py](c:\Users\Musharraf\Documents\POC\model_training\recommend_v2.py)

### 2.2 Clarity Inference

Clarity is inferred in two stages:

1. exact prompt-log lookup
- first tries `prompt_logs` from Supabase
- then tries local [prompt_logs_rows.csv](c:\Users\Musharraf\Documents\POC\model_training\prompt_logs_rows.csv)

2. heuristic fallback
- extremely short prompts -> `UNCLEAR`
- ambiguous phrases like “fix this”, “do this”, “make it better” -> `UNCLEAR`
- explicit tasks with constraints -> `CLEAR`
- partial task phrasing -> `PARTIAL`

Possible outputs:
- `CLEAR`
- `PARTIAL`
- `UNCLEAR`

Why clarity matters:
- it prevents clear prompts from being compared against unclear prompts
- this produces tighter slices and more trustworthy recommendations

Files:
- [recommender.py](c:\Users\Musharraf\Documents\POC\backend\services\recommender.py)

---

**3. Benchmark Data Access Layer**

The recommender needs benchmark rows with:
- `use_case`
- `prompt_complexity`
- `clarity`
- `model_id`
- `provider`
- `accuracy_score`
- `cost`
- `latency_ms`

Primary source:
- Supabase `benchmark_results`

Fallback source:
- local CSV [benchmark_results.csv](c:\Users\Musharraf\Documents\POC\model_training\benchmark_results.csv)

Why fallback exists:
- your standalone script already works from CSV
- local development should not fail just because Supabase is empty or unavailable

The backend also pages through Supabase results rather than assuming one fetch is enough.

Files:
- [supabase_client.py](c:\Users\Musharraf\Documents\POC\backend\services\supabase_client.py)
- [recommender.py](c:\Users\Musharraf\Documents\POC\backend\services\recommender.py)

---

**4. Slice Construction Layer**

This is the most important architectural change from the old system.

Instead of ranking across broad averages, the new system first builds a comparable candidate slice.

It uses a tiered filter strategy:

### Tier 1: Exact slice
Filter rows where:
- `use_case == user_use_case`
- `prompt_complexity == inferred_complexity`
- `clarity == inferred_clarity`

### Tier 2: Use case + complexity
If the exact slice has no sufficiently supported models:
- `use_case == user_use_case`
- `prompt_complexity == inferred_complexity`

### Tier 3: Use case only
If tier 2 is still too sparse:
- `use_case == user_use_case`

This fallback hierarchy ensures:
- we use the narrowest trustworthy slice first
- we degrade gracefully instead of either failing or using all data blindly

Files:
- [recommender.py](c:\Users\Musharraf\Documents\POC\backend\services\recommender.py)

---

**5. Support Filtering Layer**

After slicing, the system groups rows by model and removes weakly supported candidates.

Rule:
- `MIN_SAMPLES_PER_MODEL = 5`

Any model with fewer than 5 rows in the active slice is ignored.

Why this exists:
- avoids noisy winners from tiny sample sizes
- prevents one or two lucky runs from beating a stable model with larger support
- makes recommendations safer and more repeatable

This is one of the biggest trust improvements over the old architecture.

Files:
- [recommender.py](c:\Users\Musharraf\Documents\POC\backend\services\recommender.py)

---

**6. Model Summary Layer**

For each surviving model in the slice, the system computes robust statistics.

Per-model aggregates:
- `sample_count`
- `avg_accuracy`
- `median_accuracy`
- `median_cost`
- `median_latency_ms`

Why these choices:
- accuracy uses mean because it reflects expected performance across the slice
- cost and latency use median because they are more stable in the presence of outliers

This is better than the old system’s raw global averages because:
- it is local to the prompt slice
- it is more robust to skew
- it preserves support counts explicitly

Files:
- [recommender.py](c:\Users\Musharraf\Documents\POC\backend\services\recommender.py)

---

**7. Ranking Layer**

This is not a single blunt weighted score across all models.

It is a two-stage ranking policy.

### Stage 1: Quality shortlist
Find:
- highest `avg_accuracy` in the slice

Then create shortlist:
- every model whose `avg_accuracy >= top_accuracy - ACCURACY_TOLERANCE`

Config:
- `ACCURACY_TOLERANCE = 2.0`

Interpretation:
- if a model is within 2 points of the top-accuracy model, it is “good enough” to be considered a near-best quality option

This reflects the idea:
- don’t chase tiny quality differences if a cheaper/faster model is nearly as good

### Stage 2: Best value within shortlist
Within the shortlist:
- normalize median cost
- normalize median latency
- lower is better for both

Value score:
- `0.75 * normalized_cost`
- `0.25 * normalized_latency`

Then choose:
- highest value score
- tie-break by higher accuracy
- then by larger sample support

Why this design:
- quality is handled first by the shortlist gate
- efficiency only decides among safe-quality options
- this better matches real product behavior than a single global composite

Files:
- [recommender.py](c:\Users\Musharraf\Documents\POC\backend\services\recommender.py)

---

**8. Switching Policy Layer**

Choosing the best value model is not the same as recommending a switch.

The switching policy compares the chosen recommendation against the user’s current model.

Switch is recommended only if one of these is true:

### Rule 1: Accuracy improvement
- `accuracy_gain >= 2.0`

### Rule 2: Cost improvement with safe quality
- `cost_delta_pct <= -15.0`
- and quality loss is not material (`accuracy_gain >= -1.0`)

### Rule 3: Latency improvement with safe quality
- `latency_delta_pct <= -20.0`
- and quality loss is not material (`accuracy_gain >= -1.0`)

Otherwise:
- stay on current model

Why this matters:
- avoids unnecessary switching on tiny differences
- separates “best available model in slice” from “worth switching to right now”
- makes backend decisions more aligned with product logic

Files:
- [recommender.py](c:\Users\Musharraf\Documents\POC\backend\services\recommender.py)

---

**9. Explanation Layer**

After ranking and switch evaluation, the system generates structured explanation fields.

Returned fields include:
- inferred `complexity`
- inferred `clarity`
- `filter_level`
- `data_source`
- `recommended_model`
- `recommended_provider`
- `expected_accuracy`
- `expected_cost`
- `expected_latency`
- `accuracy_delta_pct`
- `cost_delta_pct`
- `latency_delta_pct`
- `sample_size`
- `slice_row_count`
- `models_considered`
- `switch_recommended`
- `policy_reason`
- `reason`
- `warnings`

Important detail:
- explanation is deterministic and template-driven
- the ranking decision is not delegated to an LLM

This preserves debuggability and makes results easier to trust.

Files:
- [schemas.py](c:\Users\Musharraf\Documents\POC\backend\models\schemas.py)
- [recommender.py](c:\Users\Musharraf\Documents\POC\backend\services\recommender.py)

---

**10. Frontend Presentation Layer**

The frontend now has two recommendation-related paths:

### 10.1 Options loading
`GET /api/inference/options`

This endpoint returns:
- supported use cases
- benchmark-backed model summaries for the baseline selector
- data source indicator

The frontend uses that to render:
- use-case selector
- baseline model dropdown
- baseline model preview metrics

### 10.2 Recommendation request
`POST /api/inference/recommend`

The frontend sends:
- `prompt`
- `use_case`
- `current_model`

The UI then renders:
- recommendation summary
- compared-against card
- rationale card
- warnings if current model is not matched

Files:
- [Inference.jsx](c:\Users\Musharraf\Documents\POC\frontend\src\pages\Inference.jsx)
- [RecommendationOutput.jsx](c:\Users\Musharraf\Documents\POC\frontend\src\components\RecommendationOutput.jsx)
- [inference.js](c:\Users\Musharraf\Documents\POC\frontend\src\api\inference.js)
- [inference.py](c:\Users\Musharraf\Documents\POC\backend\routers\inference.py)

---

**End-to-End Execution Sequence**

1. Frontend loads options from `/api/inference/options`
2. User selects use case and current model
3. User enters prompt and submits
4. Backend loads classifier
5. Backend infers `prompt_complexity`
6. Backend infers `clarity`
7. Backend loads benchmark rows:
- Supabase first
- CSV fallback if needed
8. Backend applies slice tiers:
- exact
- use_case + complexity
- use_case only
9. Backend groups rows by model
10. Backend removes low-support models
11. Backend computes summary statistics
12. Backend builds quality shortlist
13. Backend selects best value model
14. Backend compares against current model
15. Backend applies switching thresholds
16. Backend returns structured recommendation payload
17. Frontend renders recommendation cards

---

**Why This Architecture Is Better Than the Old One**

Old architecture:
- filtered too broadly
- ignored clarity
- ignored support
- used unstable global normalization
- used a single blunt weighted score
- always tried to pick a winner

New architecture:
- slices data by actual prompt characteristics
- uses fallback tiers instead of all-or-nothing logic
- requires minimum support
- uses mean accuracy + median cost/latency
- ranks quality first, value second
- only recommends switching when the gain is meaningful
- supports local development via CSV fallback
- remains deterministic and explainable

---

**Design Philosophy**

This recommender is intentionally not “AI choosing with vibes.”

It is:
- deterministic analytics first
- explanation second
- narrow-slice benchmark matching
- conservative switching logic
- product-oriented output

That makes it easier to:
- debug
- validate
- explain to stakeholders
- tune thresholds later
- replace or augment pieces independently

---

**Core Config Values**

Current policy constants:
- `MIN_SAMPLES_PER_MODEL = 5`
- `ACCURACY_TOLERANCE = 2.0`
- `MIN_ACCURACY_GAIN = 2.0`
- `MIN_COST_IMPROVEMENT_PCT = 15.0`
- `MIN_LATENCY_IMPROVEMENT_PCT = 20.0`

These are the main tuning knobs for future iteration.

---

**Main Files in the Architecture**

Recommendation logic:
- [recommender.py](c:\Users\Musharraf\Documents\POC\backend\services\recommender.py)

Data access:
- [supabase_client.py](c:\Users\Musharraf\Documents\POC\backend\services\supabase_client.py)

API layer:
- [inference.py](c:\Users\Musharraf\Documents\POC\backend\routers\inference.py)

Response contract:
- [schemas.py](c:\Users\Musharraf\Documents\POC\backend\models\schemas.py)

Standalone prototype / training:
- [recommend_v2.py](c:\Users\Musharraf\Documents\POC\model_training\recommend_v2.py)

Classifier artifact:
- [classifier.pkl](c:\Users\Musharraf\Documents\POC\model_training\artifacts\classifier.pkl)

Frontend:
- [Inference.jsx](c:\Users\Musharraf\Documents\POC\frontend\src\pages\Inference.jsx)
- [RecommendationOutput.jsx](c:\Users\Musharraf\Documents\POC\frontend\src\components\RecommendationOutput.jsx)

---

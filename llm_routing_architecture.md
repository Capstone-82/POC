# LLM Routing System Architecture

This document provides a detailed technical breakdown of the production LLM routing system. It explains the end-to-end signal flow, the core recommendation engine mechanics, the database schema responsibilities, and an honest assessment of system strengths and limitations.

---

## 1. High-Level System Overview

The routing system evaluates incoming user prompts and dynamically selects the optimal LLM (Large Language Model) to fulfill the request. The primary goal is to maximize response quality (win-rate/accuracy) while strictly minimizing cost and latency according to policy constraints.

The pipeline comprises four distinct stages:

1. **Intake & Embedding**: The system intercepts the incoming prompt, classifies its metadata (`use_case`, `complexity`, `clarity`), and generates a vector embedding (`text-embedding-3-small`, 1536 dimensions) to capture the prompt's semantic meaning.
2. **Retrieval**: The system queries a vector database to find historically benchmarked prompts that are semantically identical or highly similar (KNN search). 
3. **Signal Fusion & Aggregation**: The routing engine aggregates two disparate signal types: historical prompt-level benchmark metrics (accuracy, cost, latency) for the retrieved neighbors, and macroscopic pairwise `win_rate` metadata for the entire `use_case` domain.
4. **Scoring & Decision**: The engine computes a weighted `value_score` for all candidate models, applies business policy thresholds (e.g., minimum accuracy gains, maximum cost increases), and selects the highest-scoring model.

---

## 2. Recommendation Flow (Core Focus)

When a `POST /api/inference/recommend` request is received, the following execution flow occurs:

### Step 1: Prompt Embedding & Cache Lookup
The system computes an MD5 hash of the raw prompt text (`prompt_hash`). It queries the `prompt_embeddings` table. On a cache hit, it retrieves the existing 1536-dim vector. On a cache miss, it calls the `text-embedding-3-small` API, stores the new vector in `prompt_embeddings`, and proceeds.

### Step 2: KNN Retrieval (Vector Search)
The system executes a Postgres RPC (`knn_search`) to find the $K$ most semantically similar past prompts within the same `use_case`. 
* **Phase 1 (Strict)**: Requests $K=20$ neighbors with a minimum cosine similarity of $0.70$.
* **Phase 2 (Fallback)**: If `<5` neighbors are found, it widens the net to $K=40$ at $0.60$ similarity.
* **Phase 3 (Desperation)**: If still $<5$, it drops the similarity threshold entirely to ensure enough data points are retrieved for aggregation.

### Step 3: Candidate Selection & Signal Aggregation
The retrieved neighbor rows (from `benchmark_results`) correspond to how specific models historically performed on similar prompts. 
* Candidates that have fewer than `MIN_MODEL_NEIGHBORS` (currently set to 1) are disqualified.
* For each valid model, the engine aggregates the neighborhood statistics: it calculates similarity-weighted accuracy representations (`fallback_accuracy`), average `syntax_pass` rates, and median historical costs/latencies.

### Step 4: Win-Rate Fusion
The engine invokes `get_model_win_rates(use_case, complexity)`. It looks up the pre-computed pairwise win-rates for the models identified in Step 3. 
* **Wait, why `win_rate`?** The system relies on pairwise LLM-as-a-Judge evaluations (e.g., "Response A vs Response B") instead of absolute 0-100 scores. Win-rates correctly capture relative human preference and eliminate absolute score compression (where models all seemingly score between 88 and 92).

### Step 5: Scoring (`value_score`)
Each model is assigned a `value_score` (between 0.0 and 1.0) using a use-case specific weighting matrix. For example, in `code-generation`:
* `win_rate`: 40%
* `syntax_pass_rate`: 20%
* `cost` (normalized): 25%
* `latency` (normalized): 10%
* `confidence` (sample size): 5%

### Step 6: Decision & Policy Reasoning
The models are ranked by `value_score` descending. The top model is compared against the `current_model` (if one was provided in the request). 
* The system calculates the `win_rate_delta`, `cost_delta_pct`, and `latency_delta_pct`.
* The system enforces policy. If the `win_rate_delta > 10%` (>0.10), the switch is recommended. If the quality is flat ($\pm$5%), but cost is $>15\%$ cheaper, the switch is recommended. 
* A `reason` string is generated (e.g., *"Switch from llama3-3-70b to gemini-2-0-flash. Win rate advantage is material at +36.6%."*).

---

## 3. Fallback Logic: The Slice Approach

If the KNN vector search physically fails or returns zero rows after all phases (e.g., an entirely alien prompt domain), the system hard-bails out of KNN and relies on **Slice Recommendations**.

1. Instead of looking for similar prompts, the system drops down to categorical filters: exact matches on `prompt_complexity` AND `clarity`. 
2. If that yields no models with sufficient data (`sample_count >= 5`), it widens the slice to just `prompt_complexity`.
3. If that fails, it uses the entire `use_case` average.
4. The models in this slice are assigned the global `win_rate` for that specific slice, and the `value_score` is computed exactly as before. 

**Trade-offs of Fallback:** 
The slice approach guarantees that a recommendation is *always* returned. However, the recommendation is blunt. It optimizes for the average expected payload of strings labeled "code-generation / low-complexity" rather than the specific nuance of "Write a binary search algorithm".

---

## 4. Signal Flow: From Raw Data to Win Rate

The pipeline that feeds the recommender is an asynchronous learning loop.

1. **Raw Generation (`benchmark_results`)**: Various LLMs generate responses to prompts. These are stored with absolute `accuracy_scores` (0-100 evaluation by a judge model). 
2. **Pairwise Contests (`pairwise_results`)**: An offline job pits two models' responses to the same prompt against each other using an LLM Judge (e.g. `nova-pro`). The judge outputs a binary `winner`.
3. **Rollup (`model_win_rates`)**: A materialized engine aggregates the pairwise winners. `win_rate` is specifically defined as `wins / decisive_matches` (ties are excluded from the denominator to amplify signal).
4. **Recommender Engine**: Reads `model_win_rates` to understand relative quality standings when serving inference requests.

Because pairwise judging forces a ranking rather than an arbitrary 0-100 grade, the `win_rate` signal provides a statistically significant delta (e.g. 80% vs 50%) that is usable by a deterministic routing algorithm.

---

## 5. Table-Level Responsibilities (Supabase)

| Table | Purpose | Pipeline Usage |
| :--- | :--- | :--- |
| `prompt_embeddings` | Caches the 1536-dimensional `text-embedding-3-small` vector representation of a prompt text. | Queried first during routing. Hit bypasses OpenAI API. Miss triggers embedding generation and table insertion. |
| `benchmark_results` | The primary data warehouse for model inference outputs. Stores the prompt, response, cost, latency, and absolute `eval_scores`. | Serves as the corpus for KNN. The `knn_search` RPC calculates vector distances against this table's `prompt_hash` references. |
| `pairwise_results` | Stores the outcome of head-to-head model evaluations (Model A vs Model B) for a specific prompt. | Populated by offline background batch processes; not read during live routing. |
| `model_win_rates` | A rolled-up aggregate view of `pairwise_results`. Stores the calculated win rate, tie rate, and sample size for every model/context slice. | Heavily queried by the Recommender Engine. Defines the primary quality metric (`win_rate`) used in the `value_score` equation. |
| `routing_log` | Telemetry storage. Records the outcome of every routing API hit, capturing the recommended model, deltas, and data source (KNN vs Slice). | Used for observability and assessing how often the router defaults to Slice fallback vs KNN. |
| `ab_test_results` | Specialized storage for shadow/A-B testing. Captures the side-by-side performance of user vs recommended routed model. | Used by engineers to mathematically prove the ROI of the router before deploying it in the critical path. |
| `prompt_logs` | Simple ingress log mapping raw prompts to use-cases and clarity. | Used for offline analysis to determine what users are asking for. |
| `model_priors` | Stores default baseline metrics for models absent real-world data. | Obsolete/Rarely used. Superceded by active `benchmark_results` slice aggregations. |

---

## 6. System Assessment

### Strengths (Production Ready)
1. **Dynamic Fallbacks**: The system acts extremely defensively. It cascades gracefully from strict semantics → loose semantics → strict categories → loose categories. It never hangs or crashes on an edge-case prompt.
2. **Signal Quality**: The migration from absolute 0-100 scoring to pairwise `win_rate` solves the evaluation compression problem. It allows the router to mathematically justify a model switch because 70% win-rate is statistically differentiable from 40%.
3. **Vector Dimension Alignment**: The vector math is utilizing `1536` dimensions, effectively integrating standard OpenAI representations over standard Postgres `vector` implementations.

### Limitations & Weaknesses (Areas for Improvement)
1. **Sparsity in KNN Matching**: The system aggressively filters models that don't have enough neighbors (`MIN_MODEL_NEIGHBORS`). If a user asks a novel prompt, the KNN search will retrieve models, but likely too few rows per specific model. The system currently gets around this by lowering the requirement to 1 neighbor, which introduces immense variance.
2. **Cost/Latency Scaling**: `value_score` normalizes cost and latency using basic linear inversion. This can aggressively penalize moderately-priced, high-quality models if a single extremely cheap, low-quality model exists in the denominator.
3. **Data Freshness**: The system relies heavily on offline jobs to populate `pairwise_results` and `model_win_rates`. If the evaluator pipeline goes offline, the `win_rate` signal goes stale, and newly added models will never be routed to.
4. **Embedding Compute Latency**: On a cache miss in `prompt_embeddings`, the router must block synchronously waiting for the OpenAI network roundtrip before it can even begin vector search.

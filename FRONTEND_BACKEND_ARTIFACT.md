# Frontend + Backend Artifact

## Scope Reviewed

- `frontend/`
- `backend/`

This artifact summarizes the current architecture, runtime flow, and especially the evaluator model logic.

## High-Level System Shape

- The frontend is a Vite + React app with two pages:
  - `Training` for benchmark runs across multiple models
  - `Inference` for model recommendation against stored benchmark data
- The backend is a FastAPI app with three routers:
  - `/api/training`
  - `/api/inference`
  - `/api/test`
- Benchmark execution fans out to Bedrock and Vertex models in parallel.
- Successful model outputs are scored by a Gemini-based evaluator pool.
- Results are stored in Supabase and streamed back to the frontend with SSE.

## Backend Architecture

### Entry Point

- `backend/main.py`
  - Creates FastAPI app
  - Enables CORS for `http://localhost:5173`
  - Registers training, inference, and test routers
  - Exposes `/health`

### Core Backend Modules

- `backend/routers/training.py`
  - Starts single-prompt and CSV benchmark jobs
  - Streams progress through SSE
  - Orchestrates end-to-end prompt processing
- `backend/routers/inference.py`
  - Evaluates a prompt for complexity/quality
  - Uses benchmark data to recommend a better model
- `backend/routers/test_models.py`
  - Lists and directly tests configured Bedrock and Vertex models
- `backend/services/bedrock.py`
  - Defines Bedrock model registry
  - Builds provider-specific request bodies
  - Extracts text/tokens/cost
  - Executes Bedrock models in parallel
- `backend/services/vertex.py`
  - Defines Vertex model registry
  - Calls Gemini models via `google-genai`
  - Computes token usage and cost
- `backend/services/evaluator.py`
  - Prompt classification logic
  - Batch response scoring logic
  - Parallel evaluation over a round-robin Gemini client pool
- `backend/services/gemini_clients.py`
  - Lazy-initialized evaluator client pool
  - Round-robin selection across available Gemini/Vertex evaluator clients
- `backend/services/model_registry.py`
  - Maps use cases to allowed model short IDs
- `backend/services/recommender.py`
  - Computes recommendation from Supabase benchmark rows
- `backend/services/supabase_client.py`
  - Saves benchmark rows and prompt logs
  - Retrieves benchmark rows
- `backend/jobs/store.py`
  - In-memory async queue store for SSE job events

## Training Flow

### Single Prompt

1. Frontend posts to `/api/training/run`
2. Backend creates `job_id`
3. Background task calls `process_prompts(...)`
4. Frontend opens `EventSource` on `/api/training/stream/{job_id}`

### CSV Upload

1. Frontend posts CSV to `/api/training/upload`
2. Backend validates:
   - `prompt` column exists
   - `clarity` column exists
   - `clarity` values are `CLEAR | PARTIAL | UNCLEAR`
3. Backend creates `job_id`
4. Background task calls `process_prompts(...)`
5. Frontend consumes SSE stream

### Per-Prompt Orchestration

`training.py::_process_one_prompt(...)` does:

1. Save prompt metadata to `prompt_logs`
2. Resolve allowed model IDs from `use_case`
3. Call Bedrock and Vertex model sets in parallel
4. Split results into successful vs failed responses
5. Batch-evaluate successful responses
6. Save scored rows to `benchmark_results`
7. Push sequential SSE progress events to frontend
8. Push `model_failed` events for empty/null responses

### Prompt-Level Parallelism

- `process_prompts(...)` launches all prompts concurrently with `asyncio.gather(...)`
- Inside each prompt, provider model calls are also parallelized
- Inside evaluation, batches are parallelized again

This means concurrency exists at 3 layers:

- across prompts
- across provider models
- across evaluator batches

## Evaluator Model Logic

This is the most important part of the current backend.

### Files Involved

- `backend/services/evaluator.py`
- `backend/services/gemini_clients.py`
- `backend/test_evaluator_clients.py`

### Evaluator Pool

`gemini_clients.py` lazily builds a client pool on first use:

- up to 3 clients from:
  - `GEMINI_API_KEY1`
  - `GEMINI_API_KEY2`
  - `GEMINI_API_KEY3`
- 1 Vertex-backed client from:
  - `GOOGLE_API_KEY`

Each pool entry has:

- `client`
- `model`
- `label`

Current configured evaluator models in the pool:

- Gemini API clients use `gemini-2.5-flash`
- Vertex client also uses `gemini-2.5-flash`

So today, the pool is effectively a round-robin set of `gemini-2.5-flash` evaluator callers from multiple credentials/endpoints.

### Round-Robin Selection

`get_client()`:

- initializes the pool once
- selects the next client using a shared counter
- is protected with a thread lock

This is used by:

- `evaluate_prompt(...)`
- `evaluate_all_responses(...)`
- `evaluate_response(...)`

### Prompt Evaluation Logic

`evaluate_prompt(prompt, evaluator_model=None)`:

- pulls the next client from the pool
- sends a strict JSON-only prompt
- asks for:
  - `prompt_complexity`: `low | mid | high`
  - `prompt_quality_score`: `0-100`

Important detail:

- the `evaluator_model` argument is accepted but not actually used
- the real model always comes from the pool entry selected by `get_client()`

### Response Evaluation Logic

`evaluate_all_responses(prompt, responses, evaluator_model=None)`:

1. Responses are sorted by response length
2. Each response is truncated to 3000 chars for batching
3. A rough token estimate is computed as `len(text) // 4`
4. Responses are packed into batches up to `MAX_TOKENS_PER_BATCH = 3000`
5. Each batch is assigned the next pool client via round robin
6. All batches are evaluated in parallel through a thread pool
7. Scores are merged back into one list

Returned score shape:

```json
[
  { "model_id": "some-model", "accuracy_score": 88 }
]
```

### Batch Scoring Prompt

The evaluator prompt in `BATCH_EVAL_SYSTEM` scores each response on:

- Correctness: 40%
- Completeness: 25%
- Depth & Quality: 20%
- Practical Usefulness: 15%

It also includes hard scoring rules such as:

- working code should score `75+`
- broken or hallucinated code should score below `30`
- clearly better responses must differ by at least `10`

### Retry / Failure Behavior

`_evaluate_batch_sync(...)`:

- retries up to `MAX_RETRIES = 3`
- exponential delays:
  - 4s
  - 8s
  - 16s
- retries only on likely quota/rate/overload errors
- if scoring fails, default score is `50`

`evaluate_all_responses(...)` also falls back to default `50` for batches that raise exceptions.

### Single Response Evaluation

`evaluate_response(...)` is a single-response wrapper over the same batch prompt format.

### Net Effect

The evaluator subsystem is designed for throughput:

- multiple API keys
- round-robin balancing
- batching by estimated size
- parallel execution per batch
- defensive fallback scoring

## Model Invocation Logic

### Use-Case Filtering

`model_registry.py` controls which models are called for each use case:

- `text-generation`
- `code-generation`
- `reasoning`

Only model short IDs listed for the chosen use case are executed.

### Bedrock Models

`bedrock.py` includes:

- Meta Llama variants
- Amazon Nova variants
- Mistral variants
- DeepSeek R1

Each model has:

- `model_id`
- `provider`
- `short_id`
- `fmt`

The `fmt` controls request/response parsing differences.

### Vertex Models

`vertex.py` includes Gemini variants such as:

- `gemini-3-1-pro`
- `gemini-3-1-flash-lite`
- `gemini-2-5-pro`
- `gemini-2-5-flash`
- `gemini-2-0-flash`
- `gemini-2-0-flash-lite`

## Recommendation Flow

### Backend Logic

`inference.py` flow:

1. `evaluate_prompt(...)` classifies prompt complexity and quality
2. `get_recommendation(...)` queries Supabase benchmark rows for matching:
   - `use_case`
   - `complexity`
3. Rows are aggregated per model
4. A composite score is calculated:
   - accuracy weight: `0.6`
   - cost weight: `0.3`
   - latency weight: `0.1`
5. Best model is compared against `current_model`

### Returned Recommendation

- `recommended_model`
- `accuracy_delta`
- `cost_delta`
- `latency_delta`
- `reason`

## Frontend Architecture

### App Structure

- `frontend/src/App.jsx`
  - routes `/training` and `/inference`
- `frontend/src/components/Navbar.jsx`
  - top navigation between the two pages

### Training Page

`frontend/src/pages/Training.jsx`:

- manages:
  - prompt complexity
  - use case
  - clarity
  - single vs CSV input mode
  - live logs
  - failed model logs
  - progress state
- starts backend jobs via:
  - `startTrainingJob(...)`
  - `startCSVTrainingJob(...)`
- consumes SSE events from backend
- renders telemetry using `LiveLog`

Related components:

- `PromptInput.jsx`
- `CSVUpload.jsx`
- `LiveLog.jsx`

### Inference Page

`frontend/src/pages/Inference.jsx`:

- takes:
  - prompt
  - use case
  - current model
- posts to `/api/inference/recommend`
- renders result through `RecommendationOutput.jsx`

## Important Observations / Mismatches

These are the key implementation realities I found.

### 1. Evaluator dropdown exists but is not wired into the active UI

- `frontend/src/components/EvaluatorDropdown.jsx` exists
- it is not used by `Training.jsx` or `Inference.jsx`

### 2. Evaluator model parameter is effectively ignored

- backend evaluator functions accept `evaluator_model`
- actual model selection always comes from `get_client()` pool entries
- current pool entries all point to `gemini-2.5-flash`

### 3. Inference frontend use-case values do not match backend enum values

Backend expects:

- `text-generation`
- `code-generation`
- `reasoning`

Inference frontend sends values derived from labels like:

- `general chat`
- `technical coding`
- `logical reasoning`
- etc.

These do not match the backend `UseCase` enum and are likely to fail request validation.

### 4. Inference frontend current model values do not match backend benchmark model IDs

Frontend dropdown uses labels like:

- `GPT-4o`
- `Gemini 2.5 Pro`
- `Claude 3.5 Sonnet`

Backend benchmark rows and recommendation logic operate on internal model IDs like:

- `gemini-2-5-pro`
- `nova-pro`
- `deepseek-r1`
- etc.

So even if inference request validation passes, current-model matching may fail and default to zero baselines.

### 5. Training page hero text says Gemini 2.5 Pro judges responses, but code uses Gemini 2.5 Flash pool

- UI copy mentions `Gemini 2.5 Pro`
- actual evaluator pool config uses `gemini-2.5-flash`

### 6. SSE stop is client-side only

- `handleStop()` closes the browser `EventSource`
- backend background processing keeps running
- there is no cancel endpoint or job cancellation in backend

### 7. Job store is in-memory

- `jobs/store.py` keeps queues in process memory
- restart or multi-instance deployment would break active job tracking

### 8. Supabase async wrappers are not truly asynchronous

- `save_row`, `save_prompt_log`, and `get_benchmark_data` are `async def`
- they call synchronous `.execute()` methods directly
- this works functionally but can still block the event loop

## Mental Model of the System

- Training mode is the benchmark generator.
- Backend fans prompts out to many candidate models.
- Evaluator pool scores the returned outputs.
- Supabase becomes the benchmark memory.
- Inference mode is the benchmark consumer.
- It first classifies the incoming prompt, then recommends a model from stored benchmark stats.

## Suggested Next-Step Areas

If we continue from here, the most likely high-value areas are:

- wire evaluator selection properly, if configurable evaluation is desired
- fix inference frontend value mismatches with backend enums/model IDs
- add backend job cancellation support
- move blocking Supabase operations off the event loop
- make evaluator/UI copy consistent with actual runtime model usage

## Files Most Important For Future Work

- `backend/routers/training.py`
- `backend/services/evaluator.py`
- `backend/services/gemini_clients.py`
- `backend/services/model_registry.py`
- `backend/services/recommender.py`
- `frontend/src/pages/Training.jsx`
- `frontend/src/pages/Inference.jsx`
- `frontend/src/components/EvaluatorDropdown.jsx`

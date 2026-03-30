# Backend Specification

## Stack

- FastAPI application in [backend/main.py](c:\Users\Musharraf\Documents\POC\backend\main.py)
- Supabase for benchmark and prompt-log persistence
- AWS Bedrock for benchmarked foundation model calls
- Google Vertex AI for benchmarked Gemini-family calls and evaluator routing
- OpenAI Chat Completions API for the clarity labeling pipeline

## Runtime Entry Point

The backend starts in [backend/main.py](c:\Users\Musharraf\Documents\POC\backend\main.py).

Mounted routers:

- `/api/training`
- `/api/inference`
- `/api/test`
- `/api/clarity`

Additional route:

- `GET /health`

## Environment Variables

The current code relies on these environment variables.

### Required for Supabase

- `SUPABASE_URL`
- `SUPABASE_KEY`

### Required for AWS Bedrock

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `AWS_ACCOUNT_ID`

`AWS_ACCOUNT_ID` is used to build Meta inference-profile ARNs for some Llama models.

### Required for Vertex AI

- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_API_KEY`

The code uses the `google-genai` SDK with `vertexai=True`.

### Required for clarity labeling

- `OPENAI_API_KEY`
- `OPENAI_CLARITY_MODEL` optional, defaults to `gpt-4.1`

### Optional tuning

- `HOST`
- `PORT`
- `CLARITY_BATCH_DELAY_MS`
- `VERTEX_EVAL_GLOBAL_WEIGHT`
- `VERTEX_EVAL_REGION_WEIGHT`

## Router Overview

### Training router

File: [backend/routers/training.py](c:\Users\Musharraf\Documents\POC\backend\routers\training.py)

Endpoints:

- `POST /api/training/run`
- `POST /api/training/upload`
- `POST /api/training/upload-multi`
- `GET /api/training/stream/{job_id}`

#### `POST /api/training/run`

Accepts JSON matching `SinglePromptRequest`:

```json
{
  "prompt": "Write a product launch email",
  "prompt_complexity": "mid",
  "use_case": "text-generation",
  "clarity": "CLEAR"
}
```

Returns:

```json
{ "job_id": "..." }
```

#### `POST /api/training/upload`

Accepts `multipart/form-data`:

- `file`
- `prompt_complexity`
- `use_case`

The uploaded CSV must contain:

- `prompt`
- `clarity`

#### `POST /api/training/upload-multi`

Accepts `multipart/form-data`:

- `files`
- `prompt_complexity`
- `use_case`
- `delay_ms`

This route processes multiple CSV files sequentially and inserts an optional wait between files.

#### `GET /api/training/stream/{job_id}`

Streams SSE events from the in-memory job queue.

Observed event types include:

- `progress`
- `model_failed`
- `file_started`
- `file_done`
- `file_delay`
- `done`
- `error`

### Inference router

File: [backend/routers/inference.py](c:\Users\Musharraf\Documents\POC\backend\routers\inference.py)

Endpoints:

- `GET /api/inference/options`
- `POST /api/inference/recommend`

#### `GET /api/inference/options`

Returns:

- `data_source`
- curated use case metadata
- benchmark-backed model summaries

This endpoint is used to populate the inference UI before the user submits a prompt.

#### `POST /api/inference/recommend`

Accepts:

```json
{
  "prompt": "Design a fault-tolerant event processing system",
  "use_case": "reasoning",
  "current_model": "gemini-2-5-pro"
}
```

Returns a rich `InferenceResponse` including:

- complexity and clarity signals
- where those signals came from
- active filter tier
- selected recommendation
- baseline comparison deltas
- whether a switch is recommended
- policy rationale
- top candidate models
- warnings

### Clarity router

File: [backend/routers/clarity.py](c:\Users\Musharraf\Documents\POC\backend\routers\clarity.py)

Endpoints:

- `POST /api/clarity/upload`
- `GET /api/clarity/stream/{job_id}`
- `GET /api/clarity/download/{job_id}/{file_name}`
- `GET /api/clarity/download-zip/{job_id}`

The clarity router:

- reads prompts from a CSV
- batches them in groups of 5
- calls OpenAI for `CLEAR` / `PARTIAL` / `UNCLEAR`
- writes chunk CSVs into `backend/generated_clarity_chunks/<job_id>/`
- optionally forwards chunks into the training pipeline

### Test router

File: [backend/routers/test_models.py](c:\Users\Musharraf\Documents\POC\backend\routers\test_models.py)

Endpoints:

- `GET /api/test/available-ids`
- `GET /api/test/models`
- `POST /api/test/model/{short_id}`
- `POST /api/test/all`
- `POST /api/test/vertex`
- `POST /api/test/debug/{short_id}`

These routes are for direct model verification and debugging. They are operational helpers, not end-user product flows.

## Schemas

File: [backend/models/schemas.py](c:\Users\Musharraf\Documents\POC\backend\models\schemas.py)

Important enums:

- `ClarityLevel`
- `UseCase`
- `PromptComplexity`

Important models:

- `SinglePromptRequest`
- `JobResponse`
- `LogEvent`
- `InferenceRequest`
- `ModelStats`
- `InferenceResponse`

`InferenceResponse` is intentionally rich because the frontend surfaces explanation and comparison context, not just a raw model name.

## Service Layer

### `services/bedrock.py`

Responsibilities:

- Holds the Bedrock model registry
- Builds provider-specific request bodies
- Extracts text and token data from varying response shapes
- Estimates usage when providers omit token metadata
- Computes approximate per-call cost from hardcoded per-1K token prices
- Executes Bedrock calls in parallel with a thread pool

### `services/vertex.py`

Responsibilities:

- Holds the Vertex model registry
- Calls Gemini-family models through `google-genai`
- Extracts prompt and candidate token counts from `usage_metadata`
- Computes approximate per-call cost
- Executes Vertex calls in parallel with a thread pool

### `services/model_registry.py`

Responsibilities:

- Maps each use case to an allowed set of model short IDs
- Prevents irrelevant models from being benchmarked for a given use case

The registry currently exposes:

- `TEXT_GENERATION_MODELS`
- `CODE_GENERATION_MODELS`
- `REASONING_MODELS`
- `get_model_ids_for_use_case`

### `services/evaluator.py`

Responsibilities:

- Evaluates prompt complexity and prompt quality
- Scores benchmarked model responses with a use-case-aware rubric
- Splits scoring into token-bounded batches
- Uses a round-robin Vertex evaluator pool with failover and cooldown

Important behaviors:

- Batch size is controlled by estimated token limits, not a fixed number of responses
- Vertex clients are distributed across multiple regions
- Retryable evaluator failures trigger client cooldown and failover

### `services/gemini_clients.py`

Responsibilities:

- Builds the evaluator client pool
- Weights the global Vertex endpoint more heavily by default
- Implements per-client cooldown tracking

### `services/clarity_classifier.py`

Responsibilities:

- Sends prompt batches to OpenAI Chat Completions
- Uses a strict JSON schema response format
- Validates that every prompt ID is returned exactly once with a valid clarity label

### `services/recommender.py`

Responsibilities:

- Loads benchmark data from Supabase with local CSV fallback
- Loads a local complexity classifier if present
- Infers clarity from exact prompt logs or heuristics
- Filters benchmark rows into progressively broader slices
- Enforces minimum sample count per model
- Builds value-aware model summaries
- Applies switch/no-switch policy thresholds
- Supplies model catalog data for the inference dropdown

Current recommendation policy constants:

- `MIN_SAMPLES_PER_MODEL = 5`
- `ACCURACY_TOLERANCE = 2.0`
- `MIN_ACCURACY_GAIN = 2.0`
- `MIN_COST_IMPROVEMENT_PCT = 15.0`
- `MIN_LATENCY_IMPROVEMENT_PCT = 20.0`

### `services/supabase_client.py`

Responsibilities:

- Insert benchmark rows into `benchmark_results`
- Insert prompt rows into `prompt_logs`
- Page through Supabase result sets with `_fetch_all`
- Filter benchmark rows by use case, complexity, and clarity

## Training Data Flow

```text
prompt input
  -> save prompt log
  -> choose allowed models from model_registry
  -> call Bedrock + Vertex in parallel
  -> keep only successful responses
  -> evaluate responses in batches
  -> save benchmark rows
  -> emit SSE progress
```

## Recommendation Data Flow

```text
prompt + use_case + current_model
  -> infer complexity
  -> infer clarity
  -> load benchmark rows
  -> exact slice by use_case + complexity + clarity
  -> fallback slice by use_case + complexity
  -> fallback slice by use_case only
  -> summarize models with enough support
  -> shortlist near-top quality models
  -> choose best value model
  -> compare against current model
  -> decide switch or stay
```

## Persistence Model

### `benchmark_results`

The current code expects at least these fields:

- `provider`
- `model_id`
- `prompt`
- `prompt_complexity`
- `use_case`
- `clarity`
- `response`
- `accuracy_score`
- `cost`
- `tokens`
- `latency_ms`

### `prompt_logs`

The current code writes:

- `prompt`
- `use_case`
- `clarity`

## Error and Operational Notes

- Training job queues are in-memory only. A backend restart drops active SSE job state.
- Recommendation falls back to local CSV files when Supabase cannot provide usable benchmark rows.
- Clarity chunk files are generated locally under `backend/generated_clarity_chunks/`.
- Some Bedrock providers do not return consistent token metadata, so token counts may be estimated.
- The evaluator currently ignores the `evaluator_model` argument and always routes through the Vertex-based evaluator pool.

## Running the Backend

From the repo root:

```bash
cd backend
uvicorn main:app --reload --port 8000
```

FastAPI docs are then available at:

- `http://localhost:8000/docs`

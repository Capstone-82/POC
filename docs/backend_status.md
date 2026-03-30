# Backend Status

## Current State

The backend is no longer a thin prototype. It now supports three production-style workflows:

- Benchmark data generation
- Benchmark-backed model recommendation
- Prompt clarity labeling and export

## What Is Working

### Core app surface

- FastAPI app startup
- CORS for the Vite frontend
- Health endpoint
- Training, inference, test, and clarity routers

### Benchmarking pipeline

- Single prompt benchmark jobs
- Single CSV benchmark jobs
- Multi-CSV queued benchmark jobs
- SSE progress streaming per job
- Use-case-specific model selection
- Bedrock and Vertex model calls in parallel
- Batched evaluator scoring
- Supabase persistence for benchmark rows and prompt logs

### Recommendation pipeline

- Recommendation options endpoint for UI catalog bootstrapping
- Prompt complexity inference from local classifier when available
- Complexity heuristic fallback
- Exact prompt-log-based clarity lookup
- Local CSV prompt-log fallback
- Heuristic clarity fallback
- Supabase benchmark loading with CSV fallback
- Slice-based recommendation policy with support thresholds
- Current-model comparison and switch/no-switch policy

### Clarity pipeline

- Upload prompt CSV
- Batch prompts in groups of 5
- Call OpenAI with strict JSON schema output
- Write downloadable chunk CSV files
- Build ZIP archives for chunk outputs
- Optional forwarding into training jobs

### Operational tooling

- Enumerate configured model catalog
- Test a single model by short ID
- Test all Bedrock models
- Test all Vertex models
- Return raw Bedrock response bodies for debugging

## Important Constraints

### In-memory job queues

Training and clarity SSE queues are stored in memory only.

Implications:

- Active jobs are lost if the backend restarts
- There is no durable job history
- Horizontal scaling would need an external queue or pub/sub layer

### Local artifact dependency for recommendation fallback

Recommendation depends on these files when Supabase is unavailable:

- `model_training/artifacts/classifier.pkl`
- `model_training/benchmark_results.csv`
- `model_training/prompt_logs_rows.csv`

### Generated clarity files are local

Clarity outputs are written to:

- `backend/generated_clarity_chunks/<job_id>/`

This is appropriate for local development, but not yet a shared storage strategy.

### Evaluator model selection mismatch

The active evaluator implementation uses the Vertex-based evaluator pool from [backend/services/gemini_clients.py](c:\Users\Musharraf\Documents\POC\backend\services\gemini_clients.py).

Implication:

- The architecture supports a generic `evaluator_model` concept in some older code paths and UI leftovers
- The active scoring path is effectively standardized on the Vertex-backed evaluator pool

## Risk Areas

- Hardcoded pricing tables must be kept current manually
- Some model providers omit token metadata, so usage and cost can be estimated rather than exact
- Frontend and backend both assume localhost URLs in local development
- The multi-file training queue depends on ordered SSE event handling in the client
- Recommendation quality depends heavily on benchmark coverage and sample counts

## Next Good Improvements

- Move SSE job state to Redis or another shared store
- Add durable result history and a job-results page
- Replace hardcoded frontend API base URLs with environment configuration
- Add automated integration tests around the main router flows
- Add schema migration docs for `benchmark_results` and `prompt_logs`
- Decide whether `EvaluatorDropdown.jsx` should be revived or removed

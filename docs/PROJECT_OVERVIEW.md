# ModelMatrix Project Overview

## Purpose

ModelMatrix is a benchmark-and-recommendation workspace for LLM routing.

It has two main jobs:

- Build benchmark data by running prompts through a curated set of AWS Bedrock and Google Vertex models
- Recommend the best-value model for a new prompt using matched historical benchmark slices instead of a single global average

It also includes a separate prompt clarity labeling pipeline that classifies raw prompts into `CLEAR`, `PARTIAL`, or `UNCLEAR` batches.

## Product Areas

### 1. Benchmarking

The benchmarking flow lives in the training UI and the `/api/training/*` backend routes.

The operator chooses:

- Use case: `text-generation`, `code-generation`, or `reasoning`
- Prompt complexity label: `low`, `mid`, or `high`
- Prompt clarity label for single-prompt mode: `CLEAR`, `PARTIAL`, or `UNCLEAR`
- Input source: one prompt, one CSV, or a queue of CSV files

For each prompt, the backend:

1. Logs the prompt and its clarity label to `prompt_logs`
2. Selects only the models allowed for the chosen use case
3. Calls Bedrock and Vertex models in parallel
4. Sends successful responses to the evaluator service in batches
5. Writes benchmark rows to Supabase
6. Streams progress events back to the frontend over SSE

### 2. Recommendation

The recommendation flow lives in the inference UI and `/api/inference/*`.

The operator provides:

- A prompt to route
- A use case
- A current baseline model

The backend then:

1. Infers prompt complexity with a local classifier when available, otherwise heuristics
2. Infers prompt clarity from exact prompt log matches when available, otherwise heuristics
3. Loads benchmark rows from Supabase, with local CSV fallback
4. Builds a narrow benchmark slice
5. Keeps only models with enough supporting rows
6. Chooses the best value model among near-top quality candidates
7. Applies switching thresholds before telling the user to switch

### 3. Clarity Labeling

The clarity flow lives in the `/clarity` page and `/api/clarity/*`.

It is designed for prompt dataset preparation:

- Upload a CSV with a `prompt` column
- Backend chunks prompts into groups of 5
- OpenAI classifies each group with a strict JSON schema
- Backend writes one `prompt_set_N.csv` per chunk
- Frontend offers per-chunk downloads and a ZIP download

The backend also supports auto-forwarding clarity-labeled chunks into the training pipeline, although the current frontend always sends `auto_forward=false`.

## Architecture Summary

```text
Frontend (React + Vite)
  Benchmark page
  Recommendation page
  Clarity page
        |
        v
Backend (FastAPI)
  /api/training
  /api/inference
  /api/clarity
  /api/test
        |
        +--> AWS Bedrock model calls
        +--> Google Vertex model calls
        +--> Gemini evaluator pool on Vertex
        +--> OpenAI clarity classifier
        +--> Supabase benchmark_results + prompt_logs
        +--> local CSV fallback from model_training/
```

## Data Stores

### Supabase tables

- `benchmark_results`
- `prompt_logs`

### Local fallback artifacts

- `model_training/artifacts/classifier.pkl`
- `model_training/benchmark_results.csv`
- `model_training/prompt_logs_rows.csv`

These local files let recommendation continue even if Supabase is unavailable or empty.

## Current Model Sources

### Bedrock

The code currently benchmarks these Bedrock-backed short IDs:

- `llama4-scout`
- `llama4-maverick`
- `llama3-3-70b`
- `llama3-2-90b`
- `llama3-1-70b`
- `nova-lite`
- `nova-pro`
- `nova-premier`
- `devstral-2`
- `ministral-3-8b`
- `ministral-3b`
- `magistral-small`
- `pixtral-large-2`
- `mistral-large`
- `mistral-small`
- `deepseek-r1`

### Vertex

The code currently benchmarks these Vertex-backed short IDs:

- `gemini-3-1-pro`
- `gemini-3-1-flash-lite`
- `gemini-2-5-pro`
- `gemini-2-5-flash`
- `gemini-2-0-flash`
- `gemini-2-0-flash-lite`

The training page UI presents use-case-specific active model counts based on the model registry:

- Text generation: 17
- Code generation: 14
- Reasoning: 12

## Main Folders

```text
backend/
  main.py
  jobs/
  models/
  routers/
  services/

frontend/
  src/
    api/
    components/
    pages/

docs/
  PROJECT_OVERVIEW.md
  BACKEND_SPEC.md
  FRONTEND_SPEC.md
  backend_status.md
  docs_analysis.md

model_training/
  artifacts/
  benchmark_results.csv
  prompt_logs_rows.csv
  recommend_v2.py
```

## Key Design Choices

- Benchmarking is explicit-label driven. Training does not infer use case, complexity, or clarity from the UI payload.
- Recommendation is slice-based. It does not rank models across one broad global aggregate.
- The evaluator is use-case-aware. It scores text generation, code generation, and reasoning with different rubrics.
- Supabase is preferred, but recommendation can continue from local CSV fallback.
- SSE is used for long-running training and clarity streams so the frontend remains responsive.

## See Also

- [BACKEND_SPEC.md](c:\Users\Musharraf\Documents\POC\docs\BACKEND_SPEC.md)
- [FRONTEND_SPEC.md](c:\Users\Musharraf\Documents\POC\docs\FRONTEND_SPEC.md)
- [backend_status.md](c:\Users\Musharraf\Documents\POC\docs\backend_status.md)
- [docs_analysis.md](c:\Users\Musharraf\Documents\POC\docs\docs_analysis.md)

# ModelMatrix

ModelMatrix is a benchmark-and-recommendation workspace for LLM routing.

It combines:

- a FastAPI backend that benchmarks curated AWS Bedrock and Google Vertex models
- a React frontend for running benchmark jobs, viewing live telemetry, and getting recommendation output
- a recommendation engine that chooses the best-value model from matched benchmark slices
- a prompt clarity labeling pipeline for dataset preparation

## What It Does

### Benchmarking

You can run:

- a single prompt
- a single CSV
- a queue of CSV files

For each prompt, the backend:

1. logs the prompt and clarity
2. selects the models allowed for the chosen use case
3. calls Bedrock and Vertex models in parallel
4. evaluates responses with a use-case-aware Gemini evaluator
5. stores benchmark rows in Supabase
6. streams progress to the frontend through SSE

### Recommendation

You can submit:

- a prompt
- a use case
- your current baseline model

The backend then:

1. infers prompt complexity
2. infers prompt clarity
3. loads benchmark rows from Supabase, with local CSV fallback
4. builds the closest supported benchmark slice
5. compares candidate models
6. recommends whether to switch or stay

### Clarity Labeling

You can upload a prompt CSV and have the system:

- classify prompts as `CLEAR`, `PARTIAL`, or `UNCLEAR`
- write chunked CSV outputs
- expose chunk downloads and ZIP export

## Tech Stack

### Frontend

- React 19
- Vite 8
- React Router 7
- Tailwind CSS
- Framer Motion

### Backend

- FastAPI
- Supabase Python client
- AWS Bedrock via `boto3`
- Google Vertex AI via `google-genai`
- OpenAI Chat Completions for clarity labeling

## Repository Layout

```text
.
|-- backend/           FastAPI app, routers, services, and schemas
|-- frontend/          React app, pages, components, and API clients
|-- docs/              project, backend, and frontend documentation
|-- model_training/    local recommendation artifacts and fallback CSV data
|-- evaluations.csv
|-- sample_clarity_prompts.csv
`-- README.md
```

## Main Product Areas

### Frontend routes

- `/training` benchmark orchestration
- `/inference` recommendation workflow
- `/clarity` prompt clarity labeling

### Backend routers

- `/api/training`
- `/api/inference`
- `/api/clarity`
- `/api/test`
- `/health`

## Local Setup

### 1. Clone and open the repo

```bash
git clone <your-repo-url>
cd POC
```

### 2. Backend setup

Create and activate a virtual environment, then install Python dependencies:

```bash
cd backend
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Create `backend/.env` with the required variables.

Minimum variables used by the current code:

```env
SUPABASE_URL=...
SUPABASE_KEY=...

AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=...

GOOGLE_CLOUD_PROJECT=...
GOOGLE_API_KEY=...

OPENAI_API_KEY=...
OPENAI_CLARITY_MODEL=gpt-4.1
```

Optional:

```env
HOST=127.0.0.1
PORT=8000
CLARITY_BATCH_DELAY_MS=1200
VERTEX_EVAL_GLOBAL_WEIGHT=6
VERTEX_EVAL_REGION_WEIGHT=1
```

Run the backend:

```bash
uvicorn main:app --reload --port 8000
```

### 3. Frontend setup

In a new terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend default URL:

- `http://localhost:5173`

Backend default URL:

- `http://localhost:8000`

## Data Dependencies

### Supabase tables

The backend expects at least:

- `benchmark_results`
- `prompt_logs`

### Local fallback files

Recommendation can still run from local artifacts if Supabase is unavailable.

Key files:

- `model_training/artifacts/classifier.pkl`
- `model_training/benchmark_results.csv`
- `model_training/prompt_logs_rows.csv`

## Current Model Sources

### AWS Bedrock

- Meta Llama family
- Amazon Nova family
- Mistral family
- DeepSeek R1

### Google Vertex

- Gemini 3.1 family
- Gemini 2.5 family
- Gemini 2.0 family

The exact live catalog is defined in:

- [bedrock.py](c:\Users\Musharraf\Documents\POC\backend\services\bedrock.py)
- [vertex.py](c:\Users\Musharraf\Documents\POC\backend\services\vertex.py)
- [model_registry.py](c:\Users\Musharraf\Documents\POC\backend\services\model_registry.py)

## Important Notes

- Training and clarity SSE job queues are in-memory only.
- Clarity chunk files are generated locally under `backend/generated_clarity_chunks/`.
- Frontend API URLs are currently hardcoded to localhost.
- Recommendation uses Supabase first, then local CSV fallback.
- Some token counts and costs are estimated when provider metadata is incomplete.

## Useful Endpoints

- `GET /health`
- `GET /api/inference/options`
- `POST /api/inference/recommend`
- `POST /api/training/run`
- `POST /api/training/upload`
- `POST /api/training/upload-multi`
- `GET /api/training/stream/{job_id}`
- `POST /api/clarity/upload`
- `GET /api/clarity/stream/{job_id}`
- `GET /api/test/models`

## Documentation

- [Project Overview](c:\Users\Musharraf\Documents\POC\docs\PROJECT_OVERVIEW.md)
- [Backend Spec](c:\Users\Musharraf\Documents\POC\docs\BACKEND_SPEC.md)
- [Frontend Spec](c:\Users\Musharraf\Documents\POC\docs\FRONTEND_SPEC.md)
- [Backend Status](c:\Users\Musharraf\Documents\POC\docs\backend_status.md)
- [Docs Notes](c:\Users\Musharraf\Documents\POC\docs\docs_analysis.md)

## Suggested Next Additions

- a `docs/SETUP.md` with Supabase SQL and environment bootstrapping
- a deployment guide
- integration tests for the main API flows
- a job-results or history page in the frontend

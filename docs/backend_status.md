# LLM Recommender — Backend Implementation Status

## What's Built

A fully working FastAPI backend that benchmarks LLMs across AWS Bedrock + GCP Vertex AI
and recommends the best model for a given prompt.

---

## Architecture

```
User Prompt
    │
    ├─► POST /api/training/run      (single prompt)
    │   POST /api/training/upload   (CSV batch)
    │       │
    │       ├─ CALL 1: Gemini evaluates prompt → complexity + quality score
    │       │
    │       ├─ CALL 2+3: All 16 models called IN PARALLEL
    │       │     ├─ 12 Bedrock models  (ThreadPoolExecutor)
    │       │     └─ 4 Vertex models    (ThreadPoolExecutor)
    │       │
    │       ├─ CALL 4: ONE batch Gemini call scores ALL responses at once
    │       │
    │       ├─ Save every row to Supabase
    │       └─ Stream SSE event per model → frontend live log
    │
    └─► POST /api/inference/recommend
            ├─ Gemini evaluates prompt → complexity + quality
            └─ Recommender queries Supabase → weighted score → best model
```

---

## Models in Scope (16 total)

### AWS Bedrock (12 models)

| Provider | short_id | Notes |
|---|---|---|
| Meta | `llama3-3-70b` | ✅ ARN inference profile |
| Meta | `llama3-2-90b` | ✅ ARN inference profile |
| Meta | `llama3-1-70b` | ✅ ARN inference profile |
| Amazon | `nova-lite` | ✅ |
| Amazon | `nova-pro` | ✅ |
| Amazon | `nova-premier` | ✅ Latest Nova |
| Mistral AI | `mistral-large` | ✅ |
| Mistral AI | `mistral-small` | ✅ |
| Mistral AI | `pixtral-large` | ✅ |
| DeepSeek | `deepseek-r1` | ✅ |
| Anthropic | `claude-3-5-sonnet` | ❌ AWS payment issue |
| Anthropic | `claude-3-haiku` | ❌ AWS payment issue |

### GCP Vertex AI (4 models)

| Provider | short_id | Model |
|---|---|---|
| Google | `gemini-2-5-pro` | gemini-2.5-pro |
| Google | `gemini-2-5-flash` | gemini-2.5-flash |
| Google | `gemini-2-0-flash` | gemini-2.0-flash |
| Google | `gemini-2-0-flash-lite` | gemini-2.0-flash-lite |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/api/training/run` | Single prompt training job → `{job_id}` |
| POST | `/api/training/upload` | CSV batch training job → `{job_id}` |
| GET | `/api/training/stream/{job_id}` | SSE live stream of progress events |
| POST | `/api/inference/recommend` | Get model recommendation |
| GET | `/api/test/models` | List all 16 configured models |
| POST | `/api/test/all` | Test all Bedrock models → pass/fail |
| POST | `/api/test/vertex` | Test all Vertex AI (Gemini) models |
| POST | `/api/test/model/{short_id}` | Test one specific model |

---

## Training Call Reduction

| Approach | Evaluator calls per prompt |
|---|---|
| Original (one call per response) | 1 + 16 = **17 calls** |
| **Current (batch scoring)** | **2 calls** (prompt eval + all responses) |

For a 50-prompt CSV: **850 calls → 100 calls**

---

## SSE Event Shapes

```json
// Progress (one per model per prompt)
{
  "type": "progress",
  "prompt_index": 1,
  "total": 5,
  "model_id": "nova-pro",
  "provider": "Amazon",
  "prompt_complexity": "low",
  "prompt_quality_score": 100,
  "accuracy_score": 91,
  "cost": 0.001249,
  "tokens": 409,
  "latency_ms": 6527
}

// Done
{ "type": "done", "prompt_index": 5, "total": 5 }

// Error
{ "type": "error", "message": "...", "prompt_index": 0, "total": 5 }
```

---

## Supabase Schema

```sql
create table benchmark_results (
  id                   uuid default gen_random_uuid() primary key,
  created_at           timestamp with time zone default now(),
  provider             text,
  model_id             text,
  prompt               text,
  prompt_complexity    text check (prompt_complexity in ('low', 'mid', 'high')),
  prompt_quality_score integer check (prompt_quality_score between 0 and 100),
  response             text,
  accuracy_score       integer check (accuracy_score between 0 and 100),
  cost                 float,
  tokens               integer,
  latency_ms           integer
);
```

---

## File Structure

```
backend/
├── .env                          ← credentials (Supabase, AWS, GCP)
├── main.py                       ← FastAPI app, CORS, router mounting
├── requirements.txt
├── sample_prompts.csv            ← 5-prompt test CSV
├── models/
│   └── schemas.py                ← Pydantic request/response models
├── jobs/
│   └── store.py                  ← asyncio.Queue per job_id for SSE
├── routers/
│   ├── training.py               ← /run, /upload, /stream + pipeline logic
│   ├── inference.py              ← /recommend
│   └── test_models.py            ← /all, /vertex, /model/{short_id}
└── services/
    ├── evaluator.py              ← Gemini batch evaluator (2 calls/prompt)
    ├── bedrock.py                ← 12 Bedrock models, parallel, per-provider body format
    ├── vertex.py                 ← 4 Gemini models via Vertex AI
    ├── supabase_client.py        ← save_row + get_benchmark_data
    └── recommender.py            ← weighted scoring (60% acc, 30% cost, 10% latency)
```

---

## Known Issues

| Issue | Status |
|---|---|
| Anthropic Claude (Bedrock) | ❌ AWS Marketplace payment required |
| Mistral tokens = 0 | 🔧 Fixed — uses `prompt_tokens`/`completion_tokens` |
| Vertex 429 on CSV | 🔧 Fixed — batch eval cuts calls from 17→2 per prompt |

---

## Tested & Confirmed Working

- ✅ `POST /api/training/upload` — CSV with 5 prompts accepted
- ✅ `GET /api/training/stream/{job_id}` — SSE events streaming correctly
- ✅ All 10 non-Anthropic Bedrock models returning responses
- ✅ SSE events contain correct metadata (model_id, accuracy, cost, tokens, latency)
- ✅ Rows saving to Supabase benchmark_results table
- ✅ Vertex AI test endpoint working

---

## What's Next

1. **Frontend** — React + Vite app (Training + Inference views)
2. **Inference endpoint** — test with real Supabase data
3. **Anthropic** — resolve AWS Marketplace subscription

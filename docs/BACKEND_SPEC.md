# Backend Specification
## Python FastAPI

---

## Setup

```bash
mkdir backend && cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install fastapi uvicorn python-multipart
pip install supabase boto3 pandas
pip install google-generativeai openai anthropic
pip install asyncio httpx python-dotenv
```

---

## Folder Structure

```
backend/
  main.py
  routers/
    training.py
    inference.py
  services/
    evaluator.py       ← calls evaluator model (Gemini/GPT/Claude)
    bedrock.py         ← calls all Bedrock models
    supabase_client.py ← Supabase read/write
    recommender.py     ← recommendation logic
  models/
    schemas.py         ← Pydantic request/response models
  jobs/
    store.py           ← in-memory job store for SSE
  .env
  requirements.txt
```

---

## Environment Variables (.env)

```env
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key

# AWS Bedrock
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=us-east-1

# Evaluator model APIs
GOOGLE_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

---

## main.py

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import training, inference

app = FastAPI(title="LLM Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(training.router, prefix="/api/training")
app.include_router(inference.router, prefix="/api/inference")

@app.get("/health")
def health():
    return {"status": "ok"}
```

---

## Schemas (models/schemas.py)

```python
from pydantic import BaseModel
from typing import Optional

# Training
class SinglePromptRequest(BaseModel):
    prompt: str
    evaluator_model: str = "gemini-2.0-flash"

class JobResponse(BaseModel):
    job_id: str

# SSE event shapes (serialized to JSON strings)
class LogEvent(BaseModel):
    type: str               # "progress" | "done" | "error"
    prompt_index: int
    total: int
    model_id: str
    provider: str
    prompt_complexity: str
    prompt_quality_score: int
    accuracy_score: int
    cost: float
    tokens: int
    latency_ms: int

# Inference
class InferenceRequest(BaseModel):
    prompt: str
    use_case: str
    current_model: str

class InferenceResponse(BaseModel):
    complexity: str
    quality_score: int
    current_model: str
    recommended_model: str
    accuracy_delta: float
    cost_delta: float
    latency_delta: int
    reason: str
```

---

## Job Store (jobs/store.py)

Simple in-memory queue for SSE. Each job_id maps to a list of events.

```python
import asyncio
from collections import defaultdict

# job_id -> asyncio.Queue
job_queues: dict[str, asyncio.Queue] = {}

def create_job(job_id: str):
    job_queues[job_id] = asyncio.Queue()

async def push_event(job_id: str, event: dict):
    if job_id in job_queues:
        await job_queues[job_id].put(event)

async def get_event(job_id: str):
    return await job_queues[job_id].get()

def close_job(job_id: str):
    if job_id in job_queues:
        del job_queues[job_id]
```

---

## Training Router (routers/training.py)

### Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/training/run` | Start training job with single prompt |
| POST | `/api/training/upload` | Start training job with CSV file |
| GET | `/api/training/stream/{job_id}` | SSE stream for live log updates |

```python
import uuid
import asyncio
import json
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
import pandas as pd
from io import StringIO

from models.schemas import SinglePromptRequest, JobResponse
from jobs.store import create_job, push_event, get_event, close_job
from services.evaluator import evaluate_prompt, evaluate_response
from services.bedrock import call_all_models
from services.supabase_client import save_row

router = APIRouter()


# ─── SINGLE PROMPT ───────────────────────────────────────────

@router.post("/run", response_model=JobResponse)
async def run_single(req: SinglePromptRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    create_job(job_id)
    background_tasks.add_task(
        process_prompts,
        prompts=[req.prompt],
        evaluator_model=req.evaluator_model,
        job_id=job_id
    )
    return {"job_id": job_id}


# ─── CSV UPLOAD ───────────────────────────────────────────────

@router.post("/upload", response_model=JobResponse)
async def run_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    evaluator_model: str = Form("gemini-2.0-flash")
):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode("utf-8")))

    if "prompt" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must have a 'prompt' column")

    prompts = df["prompt"].dropna().tolist()
    job_id = str(uuid.uuid4())
    create_job(job_id)

    background_tasks.add_task(
        process_prompts,
        prompts=prompts,
        evaluator_model=evaluator_model,
        job_id=job_id
    )
    return {"job_id": job_id}


# ─── SSE STREAM ──────────────────────────────────────────────

@router.get("/stream/{job_id}")
async def stream(job_id: str):
    async def event_generator():
        while True:
            event = await get_event(job_id)

            yield f"data: {json.dumps(event)}\n\n"

            if event.get("type") in ("done", "error"):
                close_job(job_id)
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


# ─── CORE PROCESSING LOGIC ───────────────────────────────────

async def process_prompts(prompts: list[str], evaluator_model: str, job_id: str):
    total = len(prompts)

    try:
        for i, prompt in enumerate(prompts, start=1):

            # Step 1: Evaluate the prompt itself
            prompt_eval = await evaluate_prompt(prompt, evaluator_model)
            # Returns: { prompt_complexity, prompt_quality_score }

            # Step 2: Send enriched prompt to all Bedrock models
            enriched_prompt = build_enriched_prompt(prompt, prompt_eval)
            model_results = await call_all_models(enriched_prompt)
            # Returns list of: { model_id, provider, response, cost, tokens, latency_ms }

            # Step 3: For each model response, evaluate accuracy
            for result in model_results:
                response_eval = await evaluate_response(
                    prompt=prompt,
                    response=result["response"],
                    evaluator_model=evaluator_model
                )
                # Returns: { accuracy_score }

                # Step 4: Build full row
                row = {
                    "provider": result["provider"],
                    "model_id": result["model_id"],
                    "prompt": prompt,
                    "prompt_complexity": prompt_eval["prompt_complexity"],
                    "prompt_quality_score": prompt_eval["prompt_quality_score"],
                    "response": result["response"],
                    "accuracy_score": response_eval["accuracy_score"],
                    "cost": result["cost"],
                    "tokens": result["tokens"],
                    "latency_ms": result["latency_ms"],
                }

                # Step 5: Save to Supabase
                await save_row(row)

                # Step 6: Push SSE event to frontend
                await push_event(job_id, {
                    "type": "progress",
                    "prompt_index": i,
                    "total": total,
                    "model_id": result["model_id"],
                    "provider": result["provider"],
                    "prompt_complexity": prompt_eval["prompt_complexity"],
                    "prompt_quality_score": prompt_eval["prompt_quality_score"],
                    "accuracy_score": response_eval["accuracy_score"],
                    "cost": result["cost"],
                    "tokens": result["tokens"],
                    "latency_ms": result["latency_ms"],
                })

        # All done
        await push_event(job_id, {"type": "done", "prompt_index": total, "total": total})

    except Exception as e:
        await push_event(job_id, {"type": "error", "message": str(e), "prompt_index": 0, "total": total})


def build_enriched_prompt(prompt: str, prompt_eval: dict) -> str:
    return (
        f"{prompt}\n\n"
        f"[Prompt complexity: {prompt_eval['prompt_complexity']} | "
        f"Prompt quality: {prompt_eval['prompt_quality_score']}/100]"
    )
```

---

## Inference Router (routers/inference.py)

### Endpoint

| Method | Path | Description |
|---|---|---|
| POST | `/api/inference/recommend` | Get model recommendation for a prompt |

```python
from fastapi import APIRouter
from models.schemas import InferenceRequest, InferenceResponse
from services.evaluator import evaluate_prompt
from services.recommender import get_recommendation

router = APIRouter()

@router.post("/recommend", response_model=InferenceResponse)
async def recommend(req: InferenceRequest):

    # Step 1: Classify prompt
    prompt_eval = await evaluate_prompt(req.prompt, evaluator_model="gemini-2.0-flash")

    # Step 2: Run recommendation logic against Supabase benchmark data
    result = await get_recommendation(
        use_case=req.use_case,
        complexity=prompt_eval["prompt_complexity"],
        quality_score=prompt_eval["prompt_quality_score"],
        current_model=req.current_model,
    )

    return InferenceResponse(
        complexity=prompt_eval["prompt_complexity"],
        quality_score=prompt_eval["prompt_quality_score"],
        current_model=req.current_model,
        **result
    )
```

---

## Evaluator Service (services/evaluator.py)

Two functions — one evaluates the prompt, one evaluates a response.

```python
import json
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

EVALUATOR_MODELS = {
    "gemini-2.0-flash": "models/gemini-2.0-flash",
    "gpt-4o": "gpt-4o",
    "claude-sonnet-4-6": "claude-sonnet-4-6-20250514",
}


# ─── PROMPT EVALUATION ───────────────────────────────────────

PROMPT_EVAL_SYSTEM = """
You are an evaluator. Given a prompt, return a JSON object with exactly these fields:

{
  "prompt_complexity": "low" | "mid" | "high",
  "prompt_quality_score": <integer 0-100>
}

Complexity scoring:
  low  = simple factual questions, basic tasks, one-step instructions
  mid  = multi-step tasks, moderate reasoning, some domain knowledge needed
  high = complex reasoning, expert domain, multi-constraint problems

Quality scoring bands:
  90-100 = perfectly clear, specific, unambiguous
  70-89  = mostly clear, minor gaps
  40-69  = somewhat clear, notable ambiguity
  10-39  = vague or poorly structured
  0-9    = incomprehensible

Return ONLY the JSON object. No explanation, no markdown.
"""

async def evaluate_prompt(prompt: str, evaluator_model: str = "gemini-2.0-flash") -> dict:
    if evaluator_model == "gemini-2.0-flash":
        model = genai.GenerativeModel(
            model_name=EVALUATOR_MODELS[evaluator_model],
            generation_config=genai.GenerationConfig(temperature=0)
        )
        response = model.generate_content(
            f"{PROMPT_EVAL_SYSTEM}\n\nPrompt to evaluate:\n{prompt}"
        )
        return json.loads(response.text.strip())

    # Add GPT-4o and Claude branches similarly
    raise ValueError(f"Unsupported evaluator: {evaluator_model}")


# ─── RESPONSE EVALUATION ─────────────────────────────────────

RESPONSE_EVAL_SYSTEM = """
You are an evaluator. Given a prompt and a model's response, return a JSON object:

{
  "accuracy_score": <integer 0-100>
}

Scoring bands:
  90-100 = completely correct, complete, directly addresses the prompt
  70-89  = mostly correct, minor omissions or small errors
  40-69  = partially correct, notable gaps or errors
  10-39  = mostly incorrect or off-topic
  0-9    = completely wrong or irrelevant

Return ONLY the JSON object. No explanation, no markdown.
"""

async def evaluate_response(prompt: str, response: str, evaluator_model: str = "gemini-2.0-flash") -> dict:
    if evaluator_model == "gemini-2.0-flash":
        model = genai.GenerativeModel(
            model_name=EVALUATOR_MODELS[evaluator_model],
            generation_config=genai.GenerationConfig(temperature=0)
        )
        content = (
            f"{RESPONSE_EVAL_SYSTEM}\n\n"
            f"Prompt:\n{prompt}\n\n"
            f"Response:\n{response}"
        )
        result = model.generate_content(content)
        return json.loads(result.text.strip())

    raise ValueError(f"Unsupported evaluator: {evaluator_model}")
```

---

## Bedrock Service (services/bedrock.py)

```python
import boto3
import json
import time
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-east-1"),
)

BEDROCK_MODELS = [
    {"model_id": "amazon.nova-lite-v1:0",                    "provider": "Amazon"},
    {"model_id": "amazon.nova-pro-v1:0",                     "provider": "Amazon"},
    {"model_id": "meta.llama3-3-70b-instruct-v1:0",          "provider": "Meta"},
    {"model_id": "meta.llama4-scout-17b-16e-instruct-v1:0",  "provider": "Meta"},
    {"model_id": "mistral.mistral-large-2402-v1:0",          "provider": "Mistral AI"},
    {"model_id": "mistral.mistral-small-2402-v1:0",          "provider": "Mistral AI"},
    {"model_id": "us.mistral.pixtral-large-2502-v1:0",       "provider": "Mistral AI"},
    {"model_id": "deepseek-ai/deepseek-r1-0528-maas",        "provider": "DeepSeek"},
    {"model_id": "deepseek-ai/deepseek-v3.2-maas",           "provider": "DeepSeek"},
]

def _call_single_model(model_id: str, provider: str, prompt: str) -> dict:
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
    })

    start = time.time()
    response = bedrock.invoke_model(modelId=model_id, body=body)
    latency_ms = int((time.time() - start) * 1000)

    body_json = json.loads(response["body"].read())

    # Extract response text (shape varies by provider)
    text = (
        body_json.get("content", [{}])[0].get("text")          # Anthropic/Mistral
        or body_json.get("output", {}).get("message", {})
                   .get("content", [{}])[0].get("text")        # Amazon Nova
        or body_json.get("choices", [{}])[0]
                   .get("message", {}).get("content", "")      # OpenAI-compat
        or ""
    )

    # Token and cost (basic estimates — replace with real pricing)
    input_tokens  = body_json.get("usage", {}).get("input_tokens", 0)
    output_tokens = body_json.get("usage", {}).get("output_tokens", 0)
    tokens = input_tokens + output_tokens
    cost   = round(tokens * 0.000002, 6)  # placeholder rate

    return {
        "model_id":   model_id,
        "provider":   provider,
        "response":   text,
        "tokens":     tokens,
        "cost":       cost,
        "latency_ms": latency_ms,
    }

async def call_all_models(prompt: str) -> list[dict]:
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=len(BEDROCK_MODELS)) as executor:
        futures = [
            loop.run_in_executor(
                executor,
                _call_single_model,
                m["model_id"],
                m["provider"],
                prompt
            )
            for m in BEDROCK_MODELS
        ]
        results = await asyncio.gather(*futures, return_exceptions=True)

    # Filter out failed calls, log them
    clean = []
    for r in results:
        if isinstance(r, Exception):
            print(f"Model call failed: {r}")
        else:
            clean.append(r)
    return clean
```

---

## Supabase Client (services/supabase_client.py)

```python
import os
from supabase import create_client

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

async def save_row(row: dict):
    supabase.table("benchmark_results").insert(row).execute()

async def get_benchmark_data(use_case: str = None, complexity: str = None) -> list[dict]:
    query = supabase.table("benchmark_results").select("*")
    if use_case:
        query = query.eq("use_case", use_case)
    if complexity:
        query = query.eq("prompt_complexity", complexity)
    result = query.execute()
    return result.data
```

---

## Supabase Table Setup

Run this SQL in your Supabase SQL editor:

```sql
create table benchmark_results (
  id                   uuid default gen_random_uuid() primary key,
  created_at           timestamp with time zone default now(),
  provider             text,
  model_id             text,
  prompt               text,
  prompt_complexity    text,
  prompt_quality_score integer,
  response             text,
  accuracy_score       integer,
  cost                 float,
  tokens               integer,
  latency_ms           integer
);
```

---

## Recommender Service (services/recommender.py)

For v1 this uses benchmark data from Supabase directly (aggregated averages per model).
Swap this out for LightGBM later.

```python
from services.supabase_client import get_benchmark_data

WEIGHTS = {"accuracy": 0.6, "cost": 0.3, "latency": 0.1}

async def get_recommendation(
    use_case: str,
    complexity: str,
    quality_score: int,
    current_model: str
) -> dict:

    # Pull benchmark rows for this use_case + complexity
    rows = await get_benchmark_data(use_case=use_case, complexity=complexity)

    if not rows:
        raise ValueError("Not enough benchmark data for this use case and complexity.")

    # Aggregate per model
    model_stats = {}
    for row in rows:
        mid = row["model_id"]
        if mid not in model_stats:
            model_stats[mid] = {
                "accuracy_scores": [],
                "costs": [],
                "latencies": [],
                "provider": row["provider"]
            }
        model_stats[mid]["accuracy_scores"].append(row["accuracy_score"])
        model_stats[mid]["costs"].append(row["cost"])
        model_stats[mid]["latencies"].append(row["latency_ms"])

    # Compute averages + composite score
    scored = []
    for model_id, stats in model_stats.items():
        avg_acc = sum(stats["accuracy_scores"]) / len(stats["accuracy_scores"])
        avg_cost = sum(stats["costs"]) / len(stats["costs"])
        avg_lat = sum(stats["latencies"]) / len(stats["latencies"])

        # Normalize (simple min-max across all models)
        scored.append({
            "model_id": model_id,
            "provider": stats["provider"],
            "avg_accuracy": avg_acc,
            "avg_cost": avg_cost,
            "avg_latency": avg_lat,
        })

    # Normalize and score
    max_acc  = max(s["avg_accuracy"] for s in scored)
    min_acc  = min(s["avg_accuracy"] for s in scored)
    max_cost = max(s["avg_cost"] for s in scored)
    min_cost = min(s["avg_cost"] for s in scored)
    max_lat  = max(s["avg_latency"] for s in scored)
    min_lat  = min(s["avg_latency"] for s in scored)

    def norm(val, lo, hi, invert=False):
        if hi == lo:
            return 1.0
        n = (val - lo) / (hi - lo)
        return 1 - n if invert else n

    for s in scored:
        s["composite"] = (
            WEIGHTS["accuracy"] * norm(s["avg_accuracy"], min_acc, max_acc) +
            WEIGHTS["cost"]     * norm(s["avg_cost"], min_cost, max_cost, invert=True) +
            WEIGHTS["latency"]  * norm(s["avg_latency"], min_lat, max_lat, invert=True)
        )

    scored.sort(key=lambda x: x["composite"], reverse=True)
    best = scored[0]

    # Find current model stats
    current = next((s for s in scored if s["model_id"] == current_model), None)
    if not current:
        current = {"avg_accuracy": 0, "avg_cost": 0, "avg_latency": 0}

    acc_delta     = round(best["avg_accuracy"] - current["avg_accuracy"], 1)
    cost_delta    = round(((best["avg_cost"] - current["avg_cost"]) / max(current["avg_cost"], 0.0001)) * 100, 1)
    latency_delta = int(best["avg_latency"] - current["avg_latency"])

    reason = (
        f"{complexity.capitalize()}-complexity {use_case} prompt. "
        f"{best['model_id']} outperforms {current_model} on this task type "
        f"in our benchmark data."
    )

    return {
        "recommended_model": best["model_id"],
        "accuracy_delta": acc_delta,
        "cost_delta": cost_delta,
        "latency_delta": latency_delta,
        "reason": reason,
    }
```

---

## Running the Backend

```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

API docs auto-generated at: `http://localhost:8000/docs`

---

## All Endpoints Summary

| Method | Endpoint | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{ status: "ok" }` |
| POST | `/api/training/run` | `{ prompt, evaluator_model }` | `{ job_id }` |
| POST | `/api/training/upload` | FormData: `file, evaluator_model` | `{ job_id }` |
| GET | `/api/training/stream/{job_id}` | — | SSE stream of log events |
| POST | `/api/inference/recommend` | `{ prompt, use_case, current_model }` | Recommendation object |

---

## SSE Event Shapes

```json
// Progress event (one per model per prompt)
{
  "type": "progress",
  "prompt_index": 3,
  "total": 50,
  "model_id": "amazon.nova-pro-v1:0",
  "provider": "Amazon",
  "prompt_complexity": "high",
  "prompt_quality_score": 82,
  "accuracy_score": 91,
  "cost": 0.0018,
  "tokens": 890,
  "latency_ms": 1243
}

// Done event
{
  "type": "done",
  "prompt_index": 50,
  "total": 50
}

// Error event
{
  "type": "error",
  "message": "Bedrock call failed: ThrottlingException",
  "prompt_index": 12,
  "total": 50
}
```

---

## Recommendation Response Shape

```json
{
  "complexity": "high",
  "quality_score": 78,
  "current_model": "gpt-4o",
  "recommended_model": "amazon.nova-pro-v1:0",
  "accuracy_delta": 12.4,
  "cost_delta": -34.2,
  "latency_delta": -180,
  "reason": "High-complexity reasoning prompt. amazon.nova-pro-v1:0 outperforms gpt-4o on this task type in our benchmark data."
}
```

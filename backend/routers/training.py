import uuid
import json
import asyncio
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
from io import StringIO

from models.schemas import SinglePromptRequest, JobResponse
from jobs.store import create_job, push_event, get_event, close_job
from services.evaluator import evaluate_all_responses
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
        prompts=[{
            "prompt": req.prompt,
            "prompt_complexity": req.prompt_complexity,
            "prompt_quality_score": req.prompt_quality_score,
        }],
        evaluator_model=req.evaluator_model,
        job_id=job_id,
    )
    return {"job_id": job_id}


# ─── CSV UPLOAD ──────────────────────────────────────────────

@router.post("/upload", response_model=JobResponse)
async def run_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt_complexity: str = Form("mid"),
    evaluator_model: str = Form("gemini-2.0-flash"),
):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode("utf-8")))

    if "prompt" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must have a 'prompt' column")

    has_accuracy = "accuracy" in df.columns

    prompts = []
    for _, row in df.iterrows():
        p = row.get("prompt")
        if pd.isna(p) or str(p).strip() == "":
            continue
        prompts.append({
            "prompt": str(p).strip(),
            "prompt_complexity": prompt_complexity,
            "prompt_quality_score": int(row["accuracy"]) if has_accuracy and not pd.isna(row.get("accuracy")) else 50,
        })

    if not prompts:
        raise HTTPException(status_code=400, detail="No valid prompts found in CSV")

    job_id = str(uuid.uuid4())
    create_job(job_id)

    background_tasks.add_task(
        process_prompts,
        prompts=prompts,
        evaluator_model=evaluator_model,
        job_id=job_id,
    )
    return {"job_id": job_id}


# ─── SSE STREAM ─────────────────────────────────────────────

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
        },
    )


# ─── PROCESS A SINGLE PROMPT (called in parallel) ───────────

async def _process_one_prompt(
    prompt_data: dict,
    prompt_index: int,
    total: int,
    evaluator_model: str,
    job_id: str,
):
    """
    Process one prompt end-to-end:
      1. Fire ALL models in parallel (Bedrock + Vertex)
      2. Filter null responses
      3. Evaluate all successful responses in parallel (with rate limiting)
      4. Save to DB + push SSE events
    """
    prompt = prompt_data["prompt"]
    prompt_complexity = prompt_data["prompt_complexity"]
    prompt_quality_score = prompt_data["prompt_quality_score"]

    # ── STEP 1: Fire ALL Bedrock models in parallel ──────────────────
    model_results = await call_all_models(prompt)

    # ── STEP 2: Separate successful vs failed ────────────────────────────
    successful_results = []
    failed_results = []
    for r in model_results:
        if not r["response"] or str(r["response"]).strip() == "":
            failed_results.append(r)
        else:
            successful_results.append(r)

    # ── STEP 3: Evaluate ALL successful responses in parallel ────────────
    # (rate-limited by semaphore + retry in evaluator.py)
    score_map = {}
    if successful_results:
        response_payload = [
            {"model_id": r["model_id"], "response": r["response"]}
            for r in successful_results
        ]
        accuracy_scores = await evaluate_all_responses(
            prompt=prompt,
            responses=response_payload,
            evaluator_model=evaluator_model,
        )
        score_map = {s["model_id"]: s["accuracy_score"] for s in accuracy_scores}

    # ── STEP 4: Save to DB + push SSE events ─────────────────────────────
    # Save all rows in parallel too
    save_tasks = []
    for result in successful_results:
        accuracy = score_map.get(result["model_id"], 0)
        row = {
            "provider":             result["provider"],
            "model_id":             result["model_id"],
            "prompt":               prompt,
            "prompt_complexity":    prompt_complexity,
            "prompt_quality_score": prompt_quality_score,
            "response":             result["response"],
            "accuracy_score":       accuracy,
            "cost":                 result["cost"],
            "tokens":               result["tokens"],
            "latency_ms":           result["latency_ms"],
        }
        save_tasks.append(save_row(row))

    # Fire all DB saves in parallel
    if save_tasks:
        await asyncio.gather(*save_tasks)

    # Push SSE events (sequential to maintain order for the frontend)
    for result in successful_results:
        accuracy = score_map.get(result["model_id"], 0)
        await push_event(job_id, {
            "type":                "progress",
            "prompt_index":        prompt_index,
            "total":               total,
            "model_id":            result["model_id"],
            "provider":            result["provider"],
            "prompt_complexity":   prompt_complexity,
            "prompt_quality_score": prompt_quality_score,
            "accuracy_score":      accuracy,
            "cost":                result["cost"],
            "tokens":              result["tokens"],
            "latency_ms":          result["latency_ms"],
        })

    # Flag failed responses
    for result in failed_results:
        await push_event(job_id, {
            "type":                "model_failed",
            "prompt_index":        prompt_index,
            "total":               total,
            "model_id":            result["model_id"],
            "provider":            result["provider"],
            "prompt_complexity":   prompt_complexity,
            "prompt_quality_score": prompt_quality_score,
            "accuracy_score":      0,
            "cost":                0,
            "tokens":              0,
            "latency_ms":          result["latency_ms"],
            "reason":              "Model returned null/empty response",
        })


# ─── CORE ORCHESTRATOR — ALL PROMPTS IN PARALLEL ────────────

async def process_prompts(prompts: list[dict], evaluator_model: str, job_id: str):
    """
    FULL PARALLEL pipeline:
      5 prompts × 18 models = 90 model calls fired simultaneously
      → gather responses → evaluator calls with rate limiting → save to DB

    Each prompt is processed by _process_one_prompt concurrently.
    """
    total = len(prompts)

    try:
        # Fire ALL prompts at once
        tasks = [
            _process_one_prompt(
                prompt_data=prompt_data,
                prompt_index=i,
                total=total,
                evaluator_model=evaluator_model,
                job_id=job_id,
            )
            for i, prompt_data in enumerate(prompts, start=1)
        ]

        await asyncio.gather(*tasks)

        # All done
        await push_event(job_id, {"type": "done", "prompt_index": total, "total": total})

    except Exception as e:
        await push_event(job_id, {
            "type":          "error",
            "message":       str(e),
            "prompt_index":  0,
            "total":         total,
        })

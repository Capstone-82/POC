import uuid
import json
import asyncio
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
from io import StringIO

from models.schemas import SinglePromptRequest, JobResponse
from jobs.store import create_job, push_event, get_event, close_job
from services.evaluator import evaluate_prompt, evaluate_all_responses
from services.bedrock import call_all_models
from services.vertex import call_all_vertex_models
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
        job_id=job_id,
    )
    return {"job_id": job_id}


# ─── CSV UPLOAD ──────────────────────────────────────────────

@router.post("/upload", response_model=JobResponse)
async def run_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    evaluator_model: str = Form("gemini-2.0-flash"),
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


# ─── CORE PROCESSING LOGIC ──────────────────────────────────

async def process_prompts(prompts: list[str], evaluator_model: str, job_id: str):
    total = len(prompts)

    try:
        for i, prompt in enumerate(prompts, start=1):

            # ── CALL 1: Evaluate the prompt (complexity + quality) ──────
            prompt_eval = await evaluate_prompt(prompt, evaluator_model)
            # Returns: { prompt_complexity, prompt_quality_score }

            # ── CALL 2+3: Call ALL models in parallel (Bedrock + Vertex) ─
            enriched_prompt = build_enriched_prompt(prompt, prompt_eval)
            bedrock_results, vertex_results = await asyncio.gather(
                call_all_models(enriched_prompt),
                call_all_vertex_models(enriched_prompt),
            )
            model_results = bedrock_results + vertex_results
            # Each item: { model_id, provider, response, cost, tokens, latency_ms }

            # ── CALL 4: ONE batch evaluator call scores ALL responses ─────
            # Build payload: [{model_id, response}, ...]
            response_payload = [
                {"model_id": r["model_id"], "response": r["response"]}
                for r in model_results
            ]
            accuracy_scores = await evaluate_all_responses(
                prompt=prompt,
                responses=response_payload,
                evaluator_model=evaluator_model,
            )
            # Returns: [{model_id, accuracy_score}, ...]

            # Build a lookup dict for fast access
            score_map = {s["model_id"]: s["accuracy_score"] for s in accuracy_scores}

            # ── Save each row + stream SSE event ─────────────────────────
            for result in model_results:
                accuracy = score_map.get(result["model_id"], 0)

                row = {
                    "provider":            result["provider"],
                    "model_id":            result["model_id"],
                    "prompt":              prompt,
                    "prompt_complexity":   prompt_eval["prompt_complexity"],
                    "prompt_quality_score": prompt_eval["prompt_quality_score"],
                    "response":            result["response"],
                    "accuracy_score":      accuracy,
                    "cost":                result["cost"],
                    "tokens":              result["tokens"],
                    "latency_ms":          result["latency_ms"],
                }

                await save_row(row)

                await push_event(job_id, {
                    "type":                "progress",
                    "prompt_index":        i,
                    "total":               total,
                    "model_id":            result["model_id"],
                    "provider":            result["provider"],
                    "prompt_complexity":   prompt_eval["prompt_complexity"],
                    "prompt_quality_score": prompt_eval["prompt_quality_score"],
                    "accuracy_score":      accuracy,
                    "cost":                result["cost"],
                    "tokens":              result["tokens"],
                    "latency_ms":          result["latency_ms"],
                })

        # All done
        await push_event(job_id, {"type": "done", "prompt_index": total, "total": total})

    except Exception as e:
        await push_event(job_id, {
            "type":          "error",
            "message":       str(e),
            "prompt_index":  0,
            "total":         total,
        })


def build_enriched_prompt(prompt: str, prompt_eval: dict) -> str:
    return (
        f"{prompt}\n\n"
        f"[Prompt complexity: {prompt_eval['prompt_complexity']} | "
        f"Prompt quality: {prompt_eval['prompt_quality_score']}/100]"
    )

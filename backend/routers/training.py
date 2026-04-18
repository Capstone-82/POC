import asyncio
import json
import uuid
from io import StringIO

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from jobs.store import close_job, create_job, get_event, push_event
from models.schemas import ClarityLevel, JobResponse, PromptComplexity, SinglePromptRequest, UseCase
from services.bedrock import call_all_models
from services.embedding_service import compute_prompt_hash, get_or_compute_embedding
from services.model_registry import select_rotating_models_for_prompt
from services.pairwise_pipeline import refresh_model_win_rates_for_use_cases, run_pairwise_for_prompt_hashes
from services.supabase_client import save_prompt_log, save_row, supabase

router = APIRouter()
CSV_FILE_DELAY_MS = 3000
SELECTED_MODELS_PER_PROMPT = 3
VALID_CLARITY = {"CLEAR", "PARTIAL", "UNCLEAR"}


def create_training_job(prompts: list[dict], background_tasks: BackgroundTasks | None = None) -> str:
    """Create and launch a training job from already-normalized prompt rows."""
    job_id = str(uuid.uuid4())
    create_job(job_id)

    if background_tasks is not None:
        background_tasks.add_task(
            process_prompts,
            prompts=prompts,
            job_id=job_id,
        )
    else:
        asyncio.create_task(process_prompts(prompts=prompts, job_id=job_id))

    return job_id


@router.post("/run", response_model=JobResponse)
async def run_single(req: SinglePromptRequest, background_tasks: BackgroundTasks):
    job_id = create_training_job(
        prompts=[
            {
                "prompt": req.prompt,
                "prompt_complexity": req.prompt_complexity.value,
                "use_case": req.use_case.value,
                "clarity": req.clarity.value,
            }
        ],
        background_tasks=background_tasks,
    )
    return {"job_id": job_id}


@router.post("/upload", response_model=JobResponse)
async def run_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt_complexity: str = Form("mid"),
    use_case: str = Form("text-generation"),
):
    try:
        complexity_enum = PromptComplexity(prompt_complexity)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid prompt_complexity '{prompt_complexity}'. Must be one of: low, mid, high",
        )

    try:
        use_case_enum = UseCase(use_case)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid use_case '{use_case}'. Must be one of: text-generation, reasoning, code-generation",
        )

    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode("utf-8")))

    if "prompt" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must have a 'prompt' column")
    if "clarity" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must have a 'clarity' column")

    prompts = _extract_prompt_rows(df, complexity_enum.value, use_case_enum.value)
    if not prompts:
        raise HTTPException(status_code=400, detail="No valid prompts found in CSV")

    job_id = create_training_job(prompts=prompts, background_tasks=background_tasks)
    return {"job_id": job_id}


@router.post("/upload-multi", response_model=JobResponse)
async def run_multi_csv(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    prompt_complexity: str = Form("mid"),
    use_case: str = Form("text-generation"),
    delay_ms: int = Form(CSV_FILE_DELAY_MS),
):
    try:
        complexity_enum = PromptComplexity(prompt_complexity)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid prompt_complexity '{prompt_complexity}'. Must be one of: low, mid, high",
        )

    try:
        use_case_enum = UseCase(use_case)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid use_case '{use_case}'. Must be one of: text-generation, reasoning, code-generation",
        )

    if not files:
        raise HTTPException(status_code=400, detail="At least one CSV file is required")

    file_batches = []
    total_prompts = 0

    for file in files:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        if "prompt" not in df.columns:
            raise HTTPException(status_code=400, detail=f"CSV '{file.filename}' must have a 'prompt' column")
        if "clarity" not in df.columns:
            raise HTTPException(status_code=400, detail=f"CSV '{file.filename}' must have a 'clarity' column")

        prompts = _extract_prompt_rows(df, complexity_enum.value, use_case_enum.value)
        if not prompts:
            continue

        total_prompts += len(prompts)
        file_batches.append(
            {
                "file_name": file.filename or f"file_{len(file_batches) + 1}.csv",
                "prompts": prompts,
            }
        )

    if not file_batches:
        raise HTTPException(status_code=400, detail="No valid prompts found in uploaded CSV files")

    job_id = str(uuid.uuid4())
    create_job(job_id)
    background_tasks.add_task(
        process_prompt_files,
        file_batches=file_batches,
        job_id=job_id,
        total_prompts=total_prompts,
        delay_ms=max(0, delay_ms),
    )
    return {"job_id": job_id}


def _extract_prompt_rows(df: pd.DataFrame, prompt_complexity: str, use_case: str) -> list[dict]:
    prompts = []
    for _, row in df.iterrows():
        prompt_value = row.get("prompt")
        if pd.isna(prompt_value) or str(prompt_value).strip() == "":
            continue

        clarity_val = str(row.get("clarity", "CLEAR")).strip().upper()
        if clarity_val not in VALID_CLARITY:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid clarity value '{clarity_val}'. Must be one of: CLEAR, PARTIAL, UNCLEAR",
            )

        prompts.append(
            {
                "prompt": str(prompt_value).strip(),
                "prompt_complexity": prompt_complexity,
                "use_case": use_case,
                "clarity": clarity_val,
            }
        )
    return prompts


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


async def _process_one_prompt(
    prompt_data: dict,
    prompt_index: int,
    total: int,
    job_id: str,
):
    """
    Process one prompt end-to-end:
      1. Compute prompt_hash and persist prompt-level metadata
      2. Generate embedding for prompt_embeddings
      3. Select a rotating 3-model subset for the use-case
      4. Run inference on only that subset
      5. Persist benchmark rows and stream progress
    """
    prompt = prompt_data["prompt"]
    prompt_complexity = prompt_data["prompt_complexity"]
    use_case = prompt_data["use_case"]
    clarity = prompt_data["clarity"]
    prompt_hash = compute_prompt_hash(prompt)

    await push_event(
        job_id,
        {
            "type": "prompt_step",
            "prompt_index": prompt_index,
            "total": total,
            "prompt_hash": prompt_hash,
            "use_case": use_case,
            "prompt_complexity": prompt_complexity,
            "clarity": clarity,
            "step": "hash_computed",
            "message": "Prompt hash computed.",
        },
    )

    try:
        await save_prompt_log(
            {
                "prompt_hash": prompt_hash,
                "prompt": prompt,
                "use_case": use_case,
                "clarity": clarity,
            }
        )
    except Exception as exc:
        print(f"[PROMPT_LOG ERROR] Failed to log prompt: {exc}")
    else:
        await push_event(
            job_id,
            {
                "type": "prompt_step",
                "prompt_index": prompt_index,
                "total": total,
                "prompt_hash": prompt_hash,
                "use_case": use_case,
                "prompt_complexity": prompt_complexity,
                "clarity": clarity,
                "step": "prompt_logged",
                "message": "Prompt metadata stored in prompt_logs.",
            },
        )

    try:
        await get_or_compute_embedding(prompt, supabase)
    except Exception as exc:
        print(f"[EMBEDDING ERROR] Failed to cache embedding: {exc}")
    else:
        await push_event(
            job_id,
            {
                "type": "prompt_step",
                "prompt_index": prompt_index,
                "total": total,
                "prompt_hash": prompt_hash,
                "use_case": use_case,
                "prompt_complexity": prompt_complexity,
                "clarity": clarity,
                "step": "embedding_cached",
                "message": "Prompt embedding created or loaded.",
            },
        )

    selected_model_ids = set(
        select_rotating_models_for_prompt(
            use_case=use_case,
            prompt_hash=prompt_hash,
            prompt_complexity=prompt_complexity,
            clarity=clarity,
            min_models=SELECTED_MODELS_PER_PROMPT,
            max_models=SELECTED_MODELS_PER_PROMPT,
        )
    )

    await push_event(
        job_id,
        {
            "type": "models_selected",
            "prompt_index": prompt_index,
            "total": total,
            "prompt_hash": prompt_hash,
            "use_case": use_case,
            "prompt_complexity": prompt_complexity,
            "clarity": clarity,
            "selected_models": sorted(selected_model_ids),
            "selected_model_count": len(selected_model_ids),
        },
    )

    await push_event(
        job_id,
        {
            "type": "prompt_step",
            "prompt_index": prompt_index,
            "total": total,
            "prompt_hash": prompt_hash,
            "use_case": use_case,
            "prompt_complexity": prompt_complexity,
            "clarity": clarity,
            "step": "inference_started",
            "message": "Running selected models.",
        },
    )

    model_results = await call_all_models(prompt, allowed_short_ids=selected_model_ids)

    successful_results = []
    failed_results = []
    for result in model_results:
        if not result["response"] or str(result["response"]).strip() == "":
            failed_results.append(result)
        else:
            successful_results.append(result)

    save_tasks = []
    for result in successful_results:
        row = {
            "prompt_hash": prompt_hash,
            "provider": result["provider"],
            "model_id": result["model_id"],
            "prompt": prompt,
            "prompt_complexity": prompt_complexity,
            "use_case": use_case,
            "clarity": clarity,
            "response": result["response"],
            "cost": result["cost"],
            "tokens": result["tokens"],
            "latency_ms": result["latency_ms"],
        }
        save_tasks.append(save_row(row))

    if save_tasks:
        await asyncio.gather(*save_tasks)

    await push_event(
        job_id,
        {
            "type": "prompt_step",
            "prompt_index": prompt_index,
            "total": total,
            "prompt_hash": prompt_hash,
            "use_case": use_case,
            "prompt_complexity": prompt_complexity,
            "clarity": clarity,
            "step": "benchmark_saved",
            "message": "Benchmark rows stored for selected models.",
        },
    )

    for result in successful_results:
        await push_event(
            job_id,
            {
                "type": "progress",
                "prompt_index": prompt_index,
                "total": total,
                "model_id": result["model_id"],
                "provider": result["provider"],
                "prompt_complexity": prompt_complexity,
                "use_case": use_case,
                "clarity": clarity,
                "cost": result["cost"],
                "tokens": result["tokens"],
                "latency_ms": result["latency_ms"],
                "prompt_hash": prompt_hash,
                "selected_model_count": len(selected_model_ids),
            },
        )

    for result in failed_results:
        await push_event(
            job_id,
            {
                "type": "model_failed",
                "prompt_index": prompt_index,
                "total": total,
                "model_id": result["model_id"],
                "provider": result["provider"],
                "prompt_complexity": prompt_complexity,
                "use_case": use_case,
                "clarity": clarity,
                "cost": 0,
                "tokens": 0,
                "latency_ms": result["latency_ms"],
                "reason": "Model returned null/empty response",
                "prompt_hash": prompt_hash,
                "selected_model_count": len(selected_model_ids),
            },
        )

    return {
        "prompt_hash": prompt_hash,
        "use_case": use_case,
    }


async def _process_prompt_batch(
    prompts: list[dict],
    job_id: str,
    total_prompts: int,
    start_index: int,
):
    tasks = [
        _process_one_prompt(
            prompt_data=prompt_data,
            prompt_index=start_index + i,
            total=total_prompts,
            job_id=job_id,
        )
        for i, prompt_data in enumerate(prompts, start=1)
    ]
    return await asyncio.gather(*tasks)


async def _run_post_generation_pipeline(job_id: str, prompt_results: list[dict]) -> None:
    prompt_hashes = sorted({result["prompt_hash"] for result in prompt_results if result and result.get("prompt_hash")})
    use_cases = sorted({result["use_case"] for result in prompt_results if result and result.get("use_case")})

    if not prompt_hashes:
        return

    await push_event(
        job_id,
        {
            "type": "postprocess_started",
            "stage": "pairwise",
            "prompt_count": len(prompt_hashes),
        },
    )

    pairwise_summary = await asyncio.to_thread(run_pairwise_for_prompt_hashes, prompt_hashes)
    for pair_result in pairwise_summary.get("pair_results", []):
        await push_event(
            job_id,
            {
                "type": "pairwise_result",
                **pair_result,
            },
        )
    await push_event(
        job_id,
        {
            "type": "postprocess_done",
            "stage": "pairwise",
            **{key: value for key, value in pairwise_summary.items() if key != "pair_results"},
        },
    )

    await push_event(
        job_id,
        {
            "type": "postprocess_started",
            "stage": "win_rates",
            "use_cases": use_cases,
        },
    )
    rows_written = await asyncio.to_thread(refresh_model_win_rates_for_use_cases, use_cases)
    await push_event(
        job_id,
        {
            "type": "postprocess_done",
            "stage": "win_rates",
            "rows_written": rows_written,
        },
    )


async def process_prompts(prompts: list[dict], job_id: str):
    try:
        total = len(prompts)
        prompt_results = await _process_prompt_batch(
            prompts=prompts,
            job_id=job_id,
            total_prompts=total,
            start_index=0,
        )
        await _run_post_generation_pipeline(job_id, prompt_results)
        await push_event(job_id, {"type": "done", "prompt_index": total, "total": total})
    except Exception as exc:
        await push_event(
            job_id,
            {
                "type": "error",
                "message": str(exc),
                "prompt_index": 0,
                "total": len(prompts),
            },
        )


async def process_prompt_files(
    file_batches: list[dict],
    job_id: str,
    total_prompts: int,
    delay_ms: int,
):
    processed = 0
    total_files = len(file_batches)

    try:
        for file_index, file_batch in enumerate(file_batches, start=1):
            await push_event(
                job_id,
                {
                    "type": "file_started",
                    "file_index": file_index,
                    "total_files": total_files,
                    "file_name": file_batch["file_name"],
                    "file_prompt_count": len(file_batch["prompts"]),
                    "processed_prompts": processed,
                    "total": total_prompts,
                },
            )

            prompt_results = await _process_prompt_batch(
                prompts=file_batch["prompts"],
                job_id=job_id,
                total_prompts=total_prompts,
                start_index=processed,
            )
            await _run_post_generation_pipeline(job_id, prompt_results)

            processed += len(file_batch["prompts"])

            await push_event(
                job_id,
                {
                    "type": "file_done",
                    "file_index": file_index,
                    "total_files": total_files,
                    "file_name": file_batch["file_name"],
                    "processed_prompts": processed,
                    "total": total_prompts,
                },
            )

            if file_index < total_files and delay_ms > 0:
                await push_event(
                    job_id,
                    {
                        "type": "file_delay",
                        "file_index": file_index,
                        "next_file_index": file_index + 1,
                        "delay_ms": delay_ms,
                        "processed_prompts": processed,
                        "total": total_prompts,
                    },
                )
                await asyncio.sleep(delay_ms / 1000)

        await push_event(
            job_id,
            {
                "type": "done",
                "prompt_index": processed,
                "total": total_prompts,
            },
        )
    except Exception as exc:
        await push_event(
            job_id,
            {
                "type": "error",
                "message": str(exc),
                "prompt_index": processed,
                "total": total_prompts,
            },
        )

import asyncio
import json
import os
import uuid
import zipfile
from io import StringIO

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from jobs.store import close_job, create_job, get_event, push_event
from models.schemas import JobResponse
from services.clarity_classifier import classify_prompt_batch

router = APIRouter()

CLARITY_BATCH_SIZE = 5
CLARITY_BATCH_DELAY_MS = int(os.getenv("CLARITY_BATCH_DELAY_MS", "1200"))
CLARITY_OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "generated_clarity_chunks")


def _chunk_rows(rows: list[dict], size: int) -> list[list[dict]]:
    return [rows[index:index + size] for index in range(0, len(rows), size)]


def _get_job_dir(job_id: str) -> str:
    return os.path.join(CLARITY_OUTPUT_ROOT, job_id)


def _write_chunk_csv(job_id: str, chunk_index: int, chunk_rows: list[dict]) -> str:
    job_dir = _get_job_dir(job_id)
    os.makedirs(job_dir, exist_ok=True)
    file_name = f"prompt_set_{chunk_index}.csv"
    output_path = os.path.join(job_dir, file_name)
    chunk_df = pd.DataFrame(
        [{"prompt": row["prompt"], "clarity": row["clarity"]} for row in chunk_rows]
    )
    chunk_df.to_csv(output_path, index=False)
    return file_name


def _build_zip_archive(job_id: str) -> str:
    job_dir = _get_job_dir(job_id)
    if not os.path.exists(job_dir):
        raise HTTPException(status_code=404, detail="Job output directory not found")

    chunk_files = sorted(
        file_name for file_name in os.listdir(job_dir)
        if file_name.startswith("prompt_set_") and file_name.endswith(".csv")
    )
    if not chunk_files:
        raise HTTPException(status_code=404, detail="No chunk files available for this job yet")

    zip_path = os.path.join(job_dir, "clarity_chunks.zip")
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_name in chunk_files:
            file_path = os.path.join(job_dir, file_name)
            archive.write(file_path, arcname=file_name)

    return zip_path


@router.post("/upload", response_model=JobResponse)
async def upload_csv(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    contents = await file.read()

    try:
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read CSV: {exc}") from exc

    if "prompt" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must have a 'prompt' column")

    prompts = []
    for index, row in df.iterrows():
        prompt = row.get("prompt")
        if pd.isna(prompt) or str(prompt).strip() == "":
            continue
        prompts.append(
            {
                "prompt_id": str(index + 1),
                "prompt": str(prompt).strip(),
            }
        )

    if not prompts:
        raise HTTPException(status_code=400, detail="No valid prompts found in CSV")

    job_id = str(uuid.uuid4())
    create_job(job_id)
    background_tasks.add_task(process_clarity_job, job_id=job_id, prompts=prompts)
    return {"job_id": job_id}


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


@router.get("/download/{job_id}/{file_name}")
async def download_chunk(job_id: str, file_name: str):
    if not file_name.startswith("prompt_set_") or not file_name.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file name")

    file_path = os.path.join(_get_job_dir(job_id), file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Chunk file not found")

    return FileResponse(path=file_path, media_type="text/csv", filename=file_name)


@router.get("/download-zip/{job_id}")
async def download_zip(job_id: str):
    zip_path = _build_zip_archive(job_id)
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=f"clarity_chunks_{job_id}.zip",
    )


async def process_clarity_job(job_id: str, prompts: list[dict]):
    total_prompts = len(prompts)
    batches = _chunk_rows(prompts, CLARITY_BATCH_SIZE)

    try:
        await push_event(
            job_id,
            {
                "type": "started",
                "total_prompts": total_prompts,
                "total_chunks": len(batches),
            },
        )

        processed_count = 0
        for chunk_index, batch in enumerate(batches, start=1):
            labels = await classify_prompt_batch(batch)
            labels_by_id = {item["prompt_id"]: item["clarity"] for item in labels}

            chunk_rows = []
            for item in batch:
                chunk_rows.append(
                    {
                        "prompt": item["prompt"],
                        "clarity": labels_by_id[item["prompt_id"]],
                    }
                )

            file_name = _write_chunk_csv(job_id, chunk_index, chunk_rows)
            processed_count += len(batch)

            await push_event(
                job_id,
                {
                    "type": "chunk_ready",
                    "chunk_index": chunk_index,
                    "chunk_size": len(batch),
                    "processed_prompts": processed_count,
                    "total_prompts": total_prompts,
                    "file_name": file_name,
                    "download_url": f"/api/clarity/download/{job_id}/{file_name}",
                },
            )

            if chunk_index < len(batches) and CLARITY_BATCH_DELAY_MS > 0:
                await asyncio.sleep(CLARITY_BATCH_DELAY_MS / 1000)

        await push_event(
            job_id,
            {
                "type": "done",
                "processed_prompts": processed_count,
                "total_prompts": total_prompts,
                "total_chunks": len(batches),
            },
        )

    except Exception as exc:
        await push_event(
            job_id,
            {
                "type": "error",
                "message": str(exc),
                "processed_prompts": processed_count if "processed_count" in locals() else 0,
                "total_prompts": total_prompts,
            },
        )

import asyncio

# job_id -> asyncio.Queue
job_queues: dict[str, asyncio.Queue] = {}


def create_job(job_id: str):
    """Create a new job queue for SSE streaming."""
    job_queues[job_id] = asyncio.Queue()


async def push_event(job_id: str, event: dict):
    """Push an event into the job's queue."""
    if job_id in job_queues:
        await job_queues[job_id].put(event)


async def get_event(job_id: str):
    """Block until the next event is available for this job."""
    return await job_queues[job_id].get()


def close_job(job_id: str):
    """Remove the job queue after completion."""
    if job_id in job_queues:
        del job_queues[job_id]

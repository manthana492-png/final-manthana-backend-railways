"""
Manthana — Queue Service
Job status API + Redis queue management.
"""

import os
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Manthana — Queue Manager")

# Add shared to path
import sys
sys.path.insert(0, "/app/shared")


@app.get("/health")
async def health():
    return {"service": "queue", "status": "ok"}


@app.get("/job/{job_id}/status")
async def job_status(job_id: str):
    """Get job status from Redis queue."""
    from queue_client import get_job_status

    status = get_job_status(job_id)
    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if status["status"] == "disabled":
        raise HTTPException(
            status_code=503,
            detail="Redis job queue is disabled (set USE_REDIS_QUEUE=1).",
        )
    return status


@app.get("/queue/stats")
async def queue_stats():
    """Get queue statistics."""
    from queue_client import get_queue_length, get_redis, redis_queue_enabled

    if not redis_queue_enabled():
        return {
            "queue_length": 0,
            "redis_connected": False,
            "queue_backend": "disabled",
        }
    r = get_redis()
    return {
        "queue_length": get_queue_length(),
        "redis_connected": r.ping(),
        "queue_backend": "redis",
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT_QUEUE_API", 8021))
    uvicorn.run(app, host="0.0.0.0", port=port)

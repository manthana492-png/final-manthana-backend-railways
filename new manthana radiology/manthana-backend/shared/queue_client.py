"""
Manthana — Redis Queue Client
Submit and poll async analysis jobs.
"""

import os
import json
import uuid
import logging
from typing import Optional

import redis
from rq import Queue
from rq.job import Job

logger = logging.getLogger("manthana.queue_client")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

_redis_conn = None
_queue = None


def get_redis():
    """Get or create Redis connection."""
    global _redis_conn
    if _redis_conn is None:
        _redis_conn = redis.from_url(REDIS_URL)
    return _redis_conn


def get_queue(name: str = "manthana") -> Queue:
    """Get or create the job queue."""
    global _queue
    if _queue is None:
        _queue = Queue(name, connection=get_redis())
    return _queue


def submit_job(func, *args, job_id: str = None, 
               timeout: int = 600, **kwargs) -> str:
    """Submit a job to the queue.
    
    Args:
        func: The function to execute (must be importable)
        job_id: Optional custom job ID (auto-generated if None)
        timeout: Max execution time in seconds (default 10 min)
        
    Returns:
        job_id: The unique job identifier for status polling
    """
    if job_id is None:
        job_id = str(uuid.uuid4())

    q = get_queue()
    job = q.enqueue(
        func,
        *args,
        **kwargs,
        job_id=job_id,
        job_timeout=timeout,
        result_ttl=3600,  # Keep result for 1 hour
    )
    
    logger.info(f"Job submitted: {job_id}")
    return job_id


def get_job_status(job_id: str) -> dict:
    """Get the status of a queued job.
    
    Returns:
        dict with keys: status, progress, result, error
    """
    try:
        job = Job.fetch(job_id, connection=get_redis())
    except Exception:
        return {
            "job_id": job_id,
            "status": "not_found",
            "error": f"Job {job_id} not found",
        }

    result = None
    if job.is_finished:
        result = job.result
        if isinstance(result, dict):
            pass  # Already a dict
        elif hasattr(result, "model_dump"):
            result = result.model_dump()  # Pydantic model

    return {
        "job_id": job_id,
        "status": _map_status(job.get_status()),
        "result": result,
        "error": str(job.exc_info) if job.is_failed else None,
    }


def _map_status(rq_status: str) -> str:
    """Map RQ status to our job status."""
    mapping = {
        "queued": "queued",
        "started": "processing",
        "deferred": "queued",
        "finished": "complete",
        "stopped": "failed",
        "canceled": "failed",
        "failed": "failed",
    }
    return mapping.get(rq_status, "unknown")


def get_queue_length() -> int:
    """Get number of jobs waiting in queue."""
    return len(get_queue())

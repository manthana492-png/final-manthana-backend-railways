"""
Manthana — RQ Worker Process
Processes queued analysis jobs.
"""

import os
import sys
import logging

# Add shared to path
sys.path.insert(0, "/app/shared")

from queue_client import redis_queue_enabled

from redis import Redis
from rq import Worker, Queue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | worker | %(message)s",
)
logger = logging.getLogger("manthana.worker")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")


def main():
    """Start the RQ worker process."""
    if not redis_queue_enabled():
        logger.error(
            "USE_REDIS_QUEUE is not enabled; worker exiting. "
            "Set USE_REDIS_QUEUE=1 and REDIS_URL, or use docker compose --profile queue."
        )
        raise SystemExit(1)
    redis_conn = Redis.from_url(REDIS_URL)
    
    queues = [Queue("manthana", connection=redis_conn)]
    
    logger.info("Starting Manthana worker...")
    logger.info(f"Redis: {REDIS_URL}")
    logger.info(f"Queues: {[q.name for q in queues]}")
    
    worker = Worker(queues, connection=redis_conn)
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()

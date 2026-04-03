"""
Manthana — Structured JSON Logger
Consistent logging across all services.
"""

import os
import sys
import json
import logging
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Outputs structured JSON log lines for production."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "service": getattr(record, "service", record.name),
            "message": record.getMessage(),
        }

        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include extra fields
        for key in ("job_id", "modality", "model", "duration_sec"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry)


def setup_logger(service_name: str) -> logging.Logger:
    """Create a configured logger for a service.
    
    Usage:
        from shared.logger import setup_logger
        logger = setup_logger("chest_xray")
        logger.info("Processing started", extra={"job_id": "abc123"})
    """
    logger = logging.getLogger(f"manthana.{service_name}")
    
    if logger.handlers:
        return logger  # Already configured

    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    
    log_format = os.getenv("LOG_FORMAT", "json")
    if log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            f"%(asctime)s | %(levelname)-8s | {service_name} | %(message)s"
        ))

    logger.addHandler(handler)
    return logger

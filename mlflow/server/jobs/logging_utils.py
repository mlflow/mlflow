"""Shared logging utilities for MLflow job consumers."""

import logging
import os


def configure_job_consumer_logging() -> None:
    """Configure Python logging for job consumers."""
    mlflow_log_level = os.environ.get("MLFLOW_LOGGING_LEVEL", "INFO").upper()
    logging.getLogger("huey").setLevel(mlflow_log_level)
    logging.getLogger("huey.consumer").setLevel(mlflow_log_level)
    logging.getLogger("alembic.runtime.migration").setLevel(logging.WARNING)

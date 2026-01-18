"""Shared logging utilities for MLflow job consumers."""

import logging

from mlflow.utils.logging_utils import get_mlflow_log_level


def configure_job_consumer_logging() -> None:
    """Configure Python logging for job consumers."""
    logging.getLogger("huey").setLevel(get_mlflow_log_level())
    logging.getLogger("huey.consumer").setLevel(get_mlflow_log_level())
    # Suppress noisy alembic INFO logs about database context
    logging.getLogger("alembic.runtime.migration").setLevel(logging.WARNING)

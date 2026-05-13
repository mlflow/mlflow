"""Shared logging utilities for MLflow job consumers."""

import logging

from mlflow.utils.logging_utils import get_mlflow_log_level


def configure_logging_for_jobs() -> None:
    """Configure Python logging for job consumers to reduce noise for log levels above DEBUG."""
    # Suppress noisy alembic and huey INFO logs for log levels above DEBUG
    if logging.getLevelName(get_mlflow_log_level()) > logging.DEBUG:
        logging.getLogger("alembic.runtime.migration").setLevel(logging.WARNING)
        logging.getLogger("huey").setLevel(logging.WARNING)

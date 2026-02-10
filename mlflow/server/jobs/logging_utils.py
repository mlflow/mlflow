"""Shared logging utilities for MLflow job consumers."""

import logging

from mlflow.utils.logging_utils import get_mlflow_log_level


def configure_logging_for_jobs() -> None:
    """Configure Python logging for job consumers to reduce noise for log levels above DEBUG."""
    # Suppress noisy alembic INFO logs (e.g., "Context impl SQLiteImpl", "Will assume...")
    if logging.getLevelName(get_mlflow_log_level()) > logging.DEBUG:
        logging.getLogger("alembic.runtime.migration").setLevel(logging.WARNING)

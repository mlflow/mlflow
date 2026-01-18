"""Shared logging utilities for MLflow job consumers."""

import logging
import os


def configure_job_consumer_logging() -> None:
    """Configure Python logging for job consumers.

    - Sets huey loggers to INFO level to suppress noisy DEBUG logs (unless DEBUG logging enabled)
    - Sets alembic logger to WARNING level to suppress migration logs
    """
    # Only suppress huey DEBUG logs if DEBUG logging is not explicitly enabled
    # Check MLFLOW_LOGGING_LEVEL environment variable directly
    mlflow_log_level = os.environ.get("MLFLOW_LOGGING_LEVEL", "INFO").upper()
    if mlflow_log_level != "DEBUG":
        # Suppress huey DEBUG logs (e.g., "Executing online_scoring_scheduler", "executed in X.XXs")
        # Huey logs at DEBUG level by default for task execution details
        logging.getLogger("huey").setLevel(logging.INFO)
        logging.getLogger("huey.consumer").setLevel(logging.INFO)

    # Suppress alembic INFO logs
    # (e.g., "Context impl SQLiteImpl", "Will assume non-transactional DDL")
    logging.getLogger("alembic.runtime.migration").setLevel(logging.WARNING)

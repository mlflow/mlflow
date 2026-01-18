"""Shared logging utilities for MLflow job consumers."""

import logging


def configure_job_consumer_logging() -> None:
    """Configure Python logging for job consumers.

    - Sets huey loggers to INFO level to suppress noisy DEBUG logs
    - Sets alembic logger to WARNING level to suppress migration logs
    """
    # Suppress huey DEBUG logs (e.g., "Executing online_scoring_scheduler", "executed in X.XXs")
    # Huey logs at DEBUG level by default for task execution details
    logging.getLogger("huey").setLevel(logging.INFO)
    logging.getLogger("huey.consumer").setLevel(logging.INFO)

    # Suppress alembic INFO logs
    # (e.g., "Context impl SQLiteImpl", "Will assume non-transactional DDL")
    logging.getLogger("alembic.runtime.migration").setLevel(logging.WARNING)

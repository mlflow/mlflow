"""Shared logging utilities for MLflow job consumers."""

import logging

from mlflow.utils.logging_utils import get_mlflow_log_level


def configure_alembic_logging() -> None:
    """Suppress noisy alembic INFO logs unless DEBUG logging is enabled.

    Examples of suppressed logs: "Context impl SQLiteImpl", "Will assume non-transactional DDL"
    """
    if logging.getLevelName(get_mlflow_log_level()) > logging.DEBUG:
        logging.getLogger("alembic.runtime.migration").setLevel(logging.WARNING)


def configure_job_consumer_logging() -> None:
    """Configure Python logging for job consumers."""
    log_level = get_mlflow_log_level()
    logging.getLogger("huey").setLevel(log_level)
    logging.getLogger("huey.consumer").setLevel(log_level)
    configure_alembic_logging()

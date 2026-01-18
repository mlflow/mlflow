"""Shared logging utilities for MLflow job consumers."""

import logging


class SuppressOnlineScoringFilter(logging.Filter):
    """Filter that downgrades INFO logs to DEBUG for online scoring jobs.

    Downgrades INFO -> DEBUG for logs that mention:
    - run_online_trace_scorer
    - run_online_session_scorer
    - online_scoring_scheduler
    """

    ONLINE_SCORING_PATTERNS = (
        "run_online_trace_scorer",
        "run_online_session_scorer",
        "online_scoring_scheduler",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        # Only process INFO level logs
        if record.levelno != logging.INFO:
            return True

        # Check if this is an online scoring log
        msg = record.getMessage()
        if any(pattern in msg for pattern in self.ONLINE_SCORING_PATTERNS):
            # Check if the logger would show DEBUG logs
            logger = logging.getLogger(record.name)
            if logger.getEffectiveLevel() <= logging.DEBUG:
                # Downgrade to DEBUG and show
                record.levelno = logging.DEBUG
                record.levelname = "DEBUG"
                return True
            else:
                # Suppress entirely
                return False

        return True


def configure_job_consumer_logging() -> None:
    """Configure Python logging for job consumers.

    - Adds filters to huey loggers to suppress/downgrade online scoring logs
    - Sets alembic logger to WARNING level to suppress migration logs
    """
    # Use filters to selectively downgrade online scoring logs from huey. setLevel is too
    # coarse-grained - it would suppress ALL huey logs, not just online scoring ones.
    # Filters allow pattern-matching on log messages to target specific job types.
    # Add filter to each logger since child loggers may have their own handlers
    _filter = SuppressOnlineScoringFilter()
    logging.getLogger("huey").addFilter(_filter)
    logging.getLogger("huey.consumer").addFilter(_filter)
    logging.getLogger("huey.consumer.Scheduler").addFilter(_filter)

    # Suppress alembic INFO logs
    # (e.g., "Context impl SQLiteImpl", "Will assume non-transactional DDL")
    logging.getLogger("alembic.runtime.migration").setLevel(logging.WARNING)

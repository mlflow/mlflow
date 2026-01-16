"""Shared logging filters for MLflow job consumers."""

import logging


class SuppressOnlineScoringFilter(logging.Filter):
    """Filter that downgrades INFO logs to DEBUG for online scoring jobs.

    Downgrades INFO -> DEBUG for logs that mention:
    - run_online_trace_scorer
    - run_online_session_scorer
    - online_scoring_scheduler
    - _exec_job (when called for online scoring)
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

        # Downgrade INFO to DEBUG for logs mentioning online scoring patterns
        msg = record.getMessage()
        if any(pattern in msg for pattern in self.ONLINE_SCORING_PATTERNS):
            record.levelno = logging.DEBUG
            record.levelname = "DEBUG"

        return True

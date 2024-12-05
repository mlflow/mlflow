import contextvars
import logging
from dataclasses import dataclass
from typing import Optional

from mlflow.entities import LiveSpan


@dataclass
class SpanWithToken:
    """
    A utility container to hold an MLflow span and its corresponding OpenTelemetry token.

    The token is a special object that is generated when setting a span as active within
    the Open Telemetry span context. This token is required when inactivate the span i.e.
    detaching the span from the context.
    """

    span: LiveSpan
    token: Optional[contextvars.Token] = None


class LogDemotionFilter(logging.Filter):
    def __init__(self, module: str, message: str):
        super().__init__()
        self.module = module
        self.message = message

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == self.module and self.message in record.getMessage():
            record.levelno = logging.DEBUG  # Change the log level to DEBUG
            record.levelname = "DEBUG"

            # Check the log level for the logger is debug or not
            logger = logging.getLogger(self.module)
            return logger.isEnabledFor(logging.DEBUG)
        return True


def suppress_token_detach_warning_to_debug_level():
    """
    Convert the "Failed to detach context" log raised by the OpenTelemetry logger to DEBUG
    level so that it does not show up in the user's console.
    """
    from opentelemetry.context import logger as otel_logger

    if not any(isinstance(f, LogDemotionFilter) for f in otel_logger.filters):
        log_filter = LogDemotionFilter("opentelemetry.context", "Failed to detach context")
        otel_logger.addFilter(log_filter)

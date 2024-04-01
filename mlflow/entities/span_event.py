import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from mlflow.entities._mlflow_object import _MlflowObject


@dataclass
class SpanEvent(_MlflowObject):
    """
    An event that records a specific occurrences or moments in time
    during a span, such as an exception being thrown.

    Args:
        name: Name of the event.
        timestamp:  The exact time the event occurred, measured in microseconds.
        attributes: A collection of key-value pairs representing detailed
            attributes of the event, such as the exception stack trace.
    """

    name: str
    # Use current time if not provided.
    timestamp: Optional[int] = field(default=int(time.time() * 1e6))
    attributes: Dict[str, Any] = field(default_factory=dict)


class ExceptionEvent(SpanEvent):
    """
    An event that records an error during a span.
    """

    def _get_stacktrace(error: BaseException) -> str:
        """Get the stacktrace of the parent error."""
        msg = repr(error)
        try:
            if sys.version_info < (3, 10):
                tb = traceback.format_exception(error.__class__, error, error.__traceback__)
            else:
                tb = traceback.format_exception(error)
            return (msg + "\n\n".join(tb)).strip()
        except Exception:
            return msg

    def __init__(self, exception: Exception, exception_stacktrace: Optional[str] = None):
        self.name = "exception"
        self.timestamp = int(time.time() * 1e6)
        stack_trace = exception_stacktrace or self._get_stacktrace(exception)
        self.attributes = {
            "exception.message": str(exception),
            "exception.type": exception.__class__.__name__,
            "exception.stacktrace": stack_trace,
        }

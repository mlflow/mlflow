import time
from dataclasses import dataclass, field
from typing import Dict

from opentelemetry.util.types import AttributeValue

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.utils.exception_utils import get_stacktrace


@dataclass
class SpanEvent(_MlflowObject):
    """
    An event that records a specific occurrences or moments in time
    during a span, such as an exception being thrown. Compatible with OpenTelemetry.

    Args:
        name: Name of the event.
        timestamp:  The exact time the event occurred, measured in microseconds.
            If not provided, the current time will be used.
        attributes: A collection of key-value pairs representing detailed
            attributes of the event, such as the exception stack trace.
            Attributes value must be one of ``[str, int, float, bool, bytes]``
            or a sequence of these types.
    """

    name: str
    # Use current time if not provided. We need to use default factory otherwise
    # the default value will be fixed to the build time of the class.
    timestamp: int = field(default_factory=lambda: int(time.time() * 1e6))
    attributes: Dict[str, AttributeValue] = field(default_factory=dict)

    @classmethod
    def from_exception(cls, exception: Exception):
        """Create a span event from an exception."""

        return cls(
            name="exception",
            attributes={
                "exception.message": str(exception),
                "exception.type": exception.__class__.__name__,
                "exception.stacktrace": get_stacktrace(exception),
            },
        )

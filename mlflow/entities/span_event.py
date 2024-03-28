import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from mlflow.entities._mlflow_object import _MLflowObject


@dataclass
class SpanEvent(_MLflowObject):
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

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from mlflow.entities._mlflow_object import _MLflowObject


@dataclass
class SpanEvent(_MLflowObject):
    """
    Point of time event that happened during the span.

    Args:
        name: Name of the event.
        timestamp: Point of time when the event happened in microseconds.
        attributes: Arbitrary key-value pairs of the event attributes.
    """

    name: str
    # Use current time if not provided.
    timestamp: Optional[int] = field(default=int(time.time() * 1e6))
    attributes: Dict[str, Any] = field(default_factory=dict)

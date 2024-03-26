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
    timestamp: Optional[int] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

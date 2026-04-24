from dataclasses import dataclass
from typing import Any

from mlflow.entities._mlflow_object import _MlflowObject


@dataclass
class Link(_MlflowObject):
    """
    Represents an OpenTelemetry Span Link that connects spans across traces.

    Span Links allow you to link spans that don't have a parent-child relationship,
    such as spans from different traces in multi-agent systems or distributed workflows.

    Args:
        trace_id: The MLflow trace ID (tr-xxx format) of the linked span.
        span_id: The span ID within that trace (16-character hex string).
        attributes: Optional attributes describing the link relationship.
            Values must be JSON-serializable (``str``, ``int``, ``float``,
            ``bool``, or ``None``).
    """

    trace_id: str
    span_id: str
    attributes: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Link":
        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            attributes=data.get("attributes"),
        )

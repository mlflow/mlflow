import base64
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
        trace_id: The trace ID of the linked span. Accepted formats include
            MLflow trace IDs (``tr-xxx``), v4 trace IDs (``trace:/<location>/<hex>``),
            and bare hex strings.
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

    @classmethod
    def from_otel_proto(cls, proto_link) -> "Link":
        from mlflow.tracing.utils import encode_span_id, generate_mlflow_trace_id_from_otel_trace_id
        from mlflow.tracing.utils.otlp import _decode_otel_proto_anyvalue, _otel_proto_bytes_to_id

        link_trace_id = _otel_proto_bytes_to_id(proto_link.trace_id)
        link_span_id = _otel_proto_bytes_to_id(proto_link.span_id)

        attrs = {}
        for attr in proto_link.attributes:
            value = _decode_otel_proto_anyvalue(attr.value)
            if isinstance(value, bytes):
                value = base64.b64encode(value).decode("ascii")
            attrs[attr.key] = value

        return cls(
            trace_id=generate_mlflow_trace_id_from_otel_trace_id(link_trace_id),
            span_id=encode_span_id(link_span_id),
            attributes=attrs or None,
        )

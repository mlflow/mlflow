from dataclasses import dataclass

from mlflow.entities._mlflow_object import _MlflowObject


@dataclass
class SpanContext(_MlflowObject):
    """
    Following OpenTelemetry spec, request_id (=trace_id) and span_id
    are packed into SpanContext object.
    https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/trace/api.md#spancontext

    Args:
        request_id: Unique identifier of the trace.
        span_id: Unique identifier of the span.
    """

    request_id: str
    span_id: str

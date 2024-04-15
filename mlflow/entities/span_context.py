from dataclasses import dataclass

from mlflow.entities._mlflow_object import _MlflowObject


@dataclass
class SpanContext(_MlflowObject):
    """
    A span context object. OpenTelemetry compatible but defines subset of fields.
    https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/trace/api.md#spancontext
    Args:
        trace_id: Unique identifier of the trace.
        span_id: Unique identifier of the span.
    """

    trace_id: str
    span_id: str

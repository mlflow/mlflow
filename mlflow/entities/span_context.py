from dataclasses import dataclass

from mlflow.entities._mlflow_object import _MLflowObject


@dataclass
class SpanContext(_MLflowObject):
    """
    Following OpenTelemetry spec, trace_id and span_id are packed into SpanContext object.
    This design is TBD: the motivation in the original spec is to restrict the
    access to other Span fields and also allow lighter serialization and deserialization
    for the purpose of trace propagation. However, since we don't have a clear use case for
    this, we may want to just flatten this into the Span object.
    https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/trace/api.md#spancontext

    Args:
        trace_id: Unique identifier of the trace.
        span_id: Unique identifier of the span.
    """

    trace_id: str
    span_id: str

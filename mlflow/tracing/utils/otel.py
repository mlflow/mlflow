from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.trace import SpanContext


def overwrite_trace_id(span: OTelSpan, new_trace_id: int, new_span_id: int) -> OTelSpan:
    """
    Forcibly override the trace ID of the given OpenTelemetry span.

    This is a hack to work around the limitation that OpenTelemetry does not allow users to
    directly modify the trace ID of a span. For example, when we merge a remote trace into the
    local trace, we need to update the trace ID of the remote trace to the local trace ID.

    Args:
        span: The OpenTelemetry span to update.
        new_trace_id: The new trace ID to set.
    """
    if not isinstance(new_trace_id, int) or not isinstance(new_span_id, int):
        raise ValueError(
            "new_trace_id and new_span_id must be integers. "
            "Please use mlflow.tracing.utils.decode_id() to decode the hex "
            "encoded IDs to the native integer format."
        )

    context = span.get_span_context()
    new_context = SpanContext(
        trace_id=new_trace_id,
        span_id=new_span_id,
        is_remote=context.is_remote,
        trace_flags=context.trace_flags,
        trace_state=context.trace_state,
    )
    span._context = new_context
    return span


def update_parent(span: OTelSpan, new_parent_span: OTelSpan):
    """
    Update the parent span of the given OpenTelemetry span.

    This is a hack to work around the limitation that OpenTelemetry does not allow users to
    directly modify the parent span of a span.

    Args:
        span: The OpenTelemetry span to update.
        new_parent_span: The new parent span to set.
    """
    parent_context = new_parent_span.get_span_context()
    span._parent = parent_context

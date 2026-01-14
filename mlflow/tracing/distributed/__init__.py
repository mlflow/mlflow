import logging
from contextlib import contextmanager

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

import mlflow
from mlflow import MlflowException

_logger = logging.getLogger(__name__)


def get_tracing_context_headers_for_http_request():
    """
    Get the http request headers that hold information of the tracing context.
    The trace context is serialized as the traceparent header which is defined
    in the W3C TraceContext specification.
    For details, you can refer to
    https://opentelemetry.io/docs/concepts/context-propagation/
    and
    https://www.w3.org/TR/trace-context/#traceparent-header

    Returns:
        The http request headers that hold information of the tracing context.
    """
    active_span = mlflow.get_current_active_span()
    if active_span is None:
        raise MlflowException.invalid_parameter_value(
            "'get_tracing_context_headers_for_http_request' must be called within the scope "
            "of an active span."
        )
    headers = {}
    TraceContextTextMapPropagator().inject(headers)
    return headers


@contextmanager
def set_tracing_context_from_http_request_headers(headers):
    """
    Context manager to extract the trace context from the http request headers
    and set the extracted trace context as the current trace context within the
    scope of this context manager.
    The trace context must be serialized as the 'traceparent' header which is defined
    in the W3C TraceContext specification, please see
    :py:func:`mlflow.tracing.distributed.get_tracing_context_headers_for_http_request`
    for how to get the http request headers.

    Args:
        headers: The http request headers to extract the trace context from.
    """
    from mlflow.entities.trace_info import TraceInfo, TraceState
    from mlflow.tracing.trace_manager import InMemoryTraceManager
    from mlflow.tracing.utils import generate_mlflow_trace_id_from_otel_trace_id

    token = None
    otel_trace_id = None
    trace_manager = InMemoryTraceManager.get_instance()
    try:
        headers = dict(headers)

        if "Traceparent" in headers:
            # Note: Some http server framework (e.g. flask) converts http header key
            # first letter to upper case, but `TraceContextTextMapPropagator` can't
            # recognize the key 'Traceparent', so that convert it to lower case.
            traceparent = headers.pop("Traceparent")
            headers["traceparent"] = traceparent

        if "traceparent" not in headers:
            raise MlflowException.invalid_parameter_value(
                "The http request headers do not contain the required key 'traceparent', "
                "please generate the request headers "
                "by 'mlflow.tracing.distributed.get_tracing_context_headers_for_http_request' "
                "API."
            )
        ctx = TraceContextTextMapPropagator().extract(headers)
        token = context_api.attach(ctx)

        extracted_span = trace_api.get_current_span(ctx)
        span_context = extracted_span.get_span_context()
        otel_trace_id = span_context.trace_id

        trace_id = generate_mlflow_trace_id_from_otel_trace_id(otel_trace_id)
        dummy_trace_info = TraceInfo(
            trace_id=trace_id,
            trace_location=None,
            request_time=None,
            state=TraceState.IN_PROGRESS,
        )

        trace_manager.register_trace(otel_trace_id, dummy_trace_info, is_remote_trace=True)

        yield
    finally:
        if token is not None:
            context_api.detach(token)
        if otel_trace_id is not None:
            trace_manager.pop_trace(otel_trace_id)

from contextlib import contextmanager

from opentelemetry import context as context_api
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


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
    headers = {}
    TraceContextTextMapPropagator().inject(headers)
    return headers


@contextmanager
def set_tracing_context_from_http_request_headers(headers):
    """
    Extract the trace context from the http request headers,
    and return the context manager to set the extracted trace context as the
    current trace context.
    The trace context must be serialized as the traceparent header which is defined
    in the W3C TraceContext specification, please see
    :py:func:`mlflow.tracing.distributed.get_tracing_context_headers_for_http_request`
    for how to get the http request headers.

    Args:
        headers: The http request headers to extract the trace context from.
    """
    token = None
    try:
        carrier = dict(headers)
        ctx = TraceContextTextMapPropagator().extract(carrier)
        token = context_api.attach(ctx)

        yield
    finally:
        if token is not None:
            context_api.detach(token)

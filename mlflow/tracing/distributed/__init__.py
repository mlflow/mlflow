import json
import logging
from contextlib import contextmanager

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

import mlflow
from mlflow import MlflowException

_logger = logging.getLogger(__name__)


_MLFLOW_TRACE_INFO_HTTP_HEADER_KEY = "Mlflow-Trace-Info"


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
    from mlflow.tracing.trace_manager import InMemoryTraceManager

    active_span = mlflow.get_current_active_span()
    if active_span is None:
        raise MlflowException.invalid_parameter_value(
            "'get_tracing_context_headers_for_http_request' must be called within the scope "
            "of an active span."
        )
    headers = {}
    TraceContextTextMapPropagator().inject(headers)
    trace_manager = InMemoryTraceManager.get_instance()

    with trace_manager.get_trace(active_span.trace_id) as trace:
        headers[_MLFLOW_TRACE_INFO_HTTP_HEADER_KEY] = json.dumps(trace.info.to_dict())

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
    from mlflow.entities.trace_info import TraceInfo
    from mlflow.tracing.trace_manager import InMemoryTraceManager
    from mlflow.tracing.utils import generate_mlflow_trace_id_from_otel_trace_id

    token = None
    try:
        headers = dict(headers)

        if "Traceparent" in headers:
            # Note: Some http server framework (e.g. flask) converts http header key
            # first letter to upper case, but `TraceContextTextMapPropagator` can't
            # recognize the key 'Traceparent', so that convert it to lower case.
            traceparent = headers.pop("Traceparent")
            headers["traceparent"] = traceparent

        if not ("traceparent" in headers and _MLFLOW_TRACE_INFO_HTTP_HEADER_KEY in headers):
            raise MlflowException.invalid_parameter_value(
                "The http request headers do not contain the required keys 'traceparent' and "
                f"'{_MLFLOW_TRACE_INFO_HTTP_HEADER_KEY}', please generate the request headers "
                f"by 'mlflow.tracing.distributed.get_tracing_context_headers_for_http_request' "
                f"API."
            )
        ctx = TraceContextTextMapPropagator().extract(headers)
        token = context_api.attach(ctx)

        trace_manager = InMemoryTraceManager.get_instance()
        try:
            mlflow_trace_info = TraceInfo.from_dict(
                json.loads(headers[_MLFLOW_TRACE_INFO_HTTP_HEADER_KEY])
            )
        except json.JSONDecodeError as e:
            raise MlflowException(
                f"The '{_MLFLOW_TRACE_INFO_HTTP_HEADER_KEY}' HTTP header contains malformed JSON "
                "and cannot be parsed."
            ) from e
        extracted_span = trace_api.get_current_span(ctx)
        span_context = extracted_span.get_span_context()
        otel_trace_id = span_context.trace_id

        trace_manager.register_trace(otel_trace_id, mlflow_trace_info)

        if mlflow_trace_info.trace_id != generate_mlflow_trace_id_from_otel_trace_id(otel_trace_id):
            raise MlflowException(
                "Internal error: The http headers contain mismatched W3C TraceContext information "
                "and mlflow TraceInfo."
            )

        yield
    finally:
        if token is not None:
            context_api.detach(token)

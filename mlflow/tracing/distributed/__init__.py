import logging
from contextlib import contextmanager

from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

import mlflow
from mlflow.telemetry.events import TracingContextPropagation
from mlflow.telemetry.track import record_usage_event
from mlflow.tracing.provider import get_context_api, get_current_context, get_current_otel_span

_logger = logging.getLogger(__name__)


@record_usage_event(TracingContextPropagation)
def get_tracing_context_headers_for_http_request() -> dict[str, str]:
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

    Example (client code):

    .. code-block:: python

        import mlflow
        from mlflow.tracing import get_tracing_context_headers_for_http_request

        with mlflow.start_span("client-root") as client_span:
            # Get the headers that hold information of the tracing context,
            # and send request to remote agent with the headers
            headers = get_tracing_context_headers_for_http_request()
            resp = requests.post(f"{base_url}/remote_agent_handler", headers=headers)

    Example (server handler code):

    .. code-block:: python

        import mlflow
        from flask import Flask, request
        from mlflow.tracing import set_tracing_context_from_http_request_headers

        app = Flask(__name__)


        @app.post("/agent-handler")
        def handle():
            headers = dict(request.headers)
            with set_tracing_context_from_http_request_headers(headers):
                with mlflow.start_span("server-handler") as span:
                    # call agent ...
                    span.set_attribute("key", "value")
    """
    active_span = mlflow.get_current_active_span()
    if active_span is None:
        _logger.warning(
            "No active span found for fetching the trace context from. Returning an empty header."
        )
    headers = {}
    TraceContextTextMapPropagator().inject(carrier=headers, context=get_current_context())
    return headers


@record_usage_event(TracingContextPropagation)
@contextmanager
def set_tracing_context_from_http_request_headers(headers: dict[str, str]):
    """
    Context manager to extract the trace context from the http request headers
    and set the extracted trace context as the current trace context within the
    scope of this context manager.
    The trace context must be serialized as the 'traceparent' header which is defined
    in the W3C TraceContext specification, please see
    :py:func:`mlflow.tracing.get_tracing_context_headers_for_http_request`
    for how to get the http request headers.

    Args:
        headers: The http request headers to extract the trace context from.

    Example (client code):

    .. code-block:: python

        import mlflow
        from mlflow.tracing import get_tracing_context_headers_for_http_request

        with mlflow.start_span("client-root") as client_span:
            # Get the headers that hold information of the tracing context,
            # and send request to remote agent with the headers
            headers = get_tracing_context_headers_for_http_request()
            resp = requests.post(f"{base_url}/remote_agent_handler", headers=headers)

    Example (server handler code):

    .. code-block:: python

        import mlflow
        from flask import Flask, request
        from mlflow.tracing import set_tracing_context_from_http_request_headers

        app = Flask(__name__)


        @app.post("/agent-handler")
        def handle():
            headers = dict(request.headers)
            with set_tracing_context_from_http_request_headers(headers):
                with mlflow.start_span("server-handler") as span:
                    # call agent ...
                    span.set_attribute("key", "value")
    """
    from mlflow import MlflowException
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
        token = get_context_api().attach(ctx)

        extracted_span = get_current_otel_span()
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
            get_context_api().detach(token)
        if otel_trace_id is not None:
            trace_manager.pop_trace(otel_trace_id)

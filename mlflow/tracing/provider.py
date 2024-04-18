import logging
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.util._once import Once

from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_status import TraceStatus
from mlflow.exceptions import RestException
from mlflow.tracing.clients import TraceClient, get_trace_client
from mlflow.tracing.export.mlflow import MlflowSpanExporter
from mlflow.tracing.utils import encode_trace_id
from mlflow.utils.databricks_utils import (
    is_in_databricks_model_serving_environment,
    is_in_databricks_runtime,
)

# Once() object ensures a function is executed only once in a process.
# Note that it doesn't work as expected in a distributed environment.
_TRACER_PROVIDER_INITIALIZED = Once()


_logger = logging.getLogger(__name__)


def get_tracer(module_name: str):
    """
    Get a tracer instance for the given module name.
    """
    # Initiate tracer provider only once in the application lifecycle
    _TRACER_PROVIDER_INITIALIZED.do_once(_setup_tracer_provider)

    tracer_provider = trace.get_tracer_provider()
    return tracer_provider.get_tracer(module_name)


def _setup_tracer_provider(client: Optional[TraceClient] = None):
    """
    Instantiate a tracer provider and set it as the global tracer provider.
    """
    client = client or get_trace_client()

    # TODO: Make factory method for exporters once we support more sink destinations
    exporter = MlflowSpanExporter(client)

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(tracer_provider)


def create_trace_info(
    otel_span: trace.Span,
    experiment_id: Optional[str] = None,
    request_metadata: Optional[dict] = None,
    tags: Optional[dict] = None,
) -> TraceInfo:
    """
    Create a new TraceInfo object based on the OpenTelemetry span and the given metadata.

    The TraceInfo generation logic depends on the environment where the trace is created,
    and this function is responsible for encapsulating the differences.

    Args:
        otel_span: The OpenTelemetry span object.
        experiment_id: The experiment ID for the trace.
        request_metadata: The metadata of the request.
        tags: The tags of the trace.
    """
    from mlflow.tracking.fluent import _get_experiment_id

    experiment_id = experiment_id or _get_experiment_id()
    request_metadata = request_metadata or {}
    tags = tags or {}
    # If the environment is Databricks, the trace should be logged in the tracking server.
    # The initial TraceInfo entry is created by calling StartTrace backend API.
    if is_in_databricks_runtime():
        from mlflow.tracking.client import MlflowClient

        try:
            return MlflowClient()._start_tracked_trace(
                experiment_id=experiment_id,
                timestamp_ms=otel_span.start_time // 1_000_000,  # nanosecond to millisecond
                request_metadata=request_metadata,
                # Some tags like mlflow.runName are immutable once logged in tracking server.
                tags={k: v for k, v in tags.items() if not k.startswith("mlflow.")},
            )
        # TODO: This catches all exceptions from the tracking server so the in-memory tracing still
        # works if the backend APIs are not ready. Once backend is ready, we should catch more
        # specific exceptions and handle them accordingly.
        except RestException:
            _logger.warning(
                "Failed to start a trace in the tracking server. This may be because the "
                "backend APIs are not available. Fallback to client-side generation"
            )
            pass

    # In Databricks model serving, TraceInfo is created at the client side and request_id
    # should be constructed based on the request payload.
    elif is_in_databricks_model_serving_environment():
        raise NotImplementedError("Model serving environment is not supported yet.")

    # Fallback to create a trace at the client side and set the trace ID as the request ID.
    return TraceInfo(
        request_id=encode_trace_id(otel_span.context.trace_id),
        experiment_id=experiment_id,
        timestamp_ms=otel_span.start_time // 1_000_000,  # nanosecond to millisecond
        execution_time_ms=None,
        status=TraceStatus.IN_PROGRESS,
        request_metadata=request_metadata or {},
        tags=tags or {},
    )


def start_span_in_context(name) -> trace.Span:
    """
    Start a new OpenTelemetry span in the current context.

    Note that this function doesn't set the started span as the active span in the context. To do
    that, the upstream also need to call `use_span()` function in the OpenTelemetry trace APIs.

    Args:
        name: The name of the span.

    Returns:
        The newly created OpenTelemetry span.
    """
    return get_tracer(__name__).start_span(name)


def start_detached_span(name: str, parent: Optional[trace.Span] = None) -> Optional[trace.Span]:
    """
    Start a new OpenTelemetry span that is not part of the current trace context, but with the
    explicit parent span ID if provided.

    Args:
        name: The name of the span.
        parent: The parent OpenTelemetry span. If not provided, the span will be created as a root
                span.

    Returns:
        The newly created OpenTelemetry span.
    """
    tracer = get_tracer(__name__)
    context = trace.set_span_in_context(parent) if parent else None
    return tracer.start_span(name, context=context)

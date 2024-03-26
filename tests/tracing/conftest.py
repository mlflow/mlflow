import pytest
from opentelemetry.trace import _TRACER_PROVIDER_SET_ONCE

from mlflow.entities import SpanStatus, TraceStatus, Trace, TraceInfo
from mlflow.tracing.clients.local import InMemoryTraceClient
from mlflow.tracing.provider import _TRACER_PROVIDER_INITIALIZED, _setup_tracer_provider
from mlflow.tracing.trace_manager import InMemoryTraceManager


@pytest.fixture(autouse=True)
def clear_singleton():
    """
    Clear the singleton instances after each tests to avoid side effects.
    """
    InMemoryTraceManager._instance = None
    InMemoryTraceClient._instance = None

    # Tracer provider also needs to be reset as it may hold reference to the singleton
    with _TRACER_PROVIDER_SET_ONCE._lock:
        _TRACER_PROVIDER_SET_ONCE._done = False
    with _TRACER_PROVIDER_INITIALIZED._lock:
        _TRACER_PROVIDER_INITIALIZED._done = False


@pytest.fixture
def mock_client():
    # OpenTelemetry doesn't allow re-initializing the tracer provider within a single
    # process. However, we need to create a new tracer provider with the new mock client
    # so hack the Once object to allow re-initialization.
    from opentelemetry.trace import _TRACER_PROVIDER_SET_ONCE

    with _TRACER_PROVIDER_SET_ONCE._lock:
        _TRACER_PROVIDER_SET_ONCE._done = False

    mock_client = InMemoryTraceClient.get_instance()

    _setup_tracer_provider(mock_client)

    yield mock_client

    # Clear traces collected in the buffer
    mock_client._flush()


@pytest.fixture
def create_trace():
    return lambda id: Trace(
        trace_info=TraceInfo(
            trace_id=id,
            experiment_id="test",
            start_time=0,
            end_time=1,
            status=SpanStatus(TraceStatus.OK),
            attributes={},
            tags={},
        ),
        trace_data=[],
    )

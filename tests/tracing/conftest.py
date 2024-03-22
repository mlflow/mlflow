import pytest

from mlflow.tracing.clients.local import InMemoryTraceClient
from mlflow.tracing.export.mlflow import InMemoryTraceDataAggregator
from mlflow.tracing.provider import _setup_tracer_provider


@pytest.fixture(autouse=True)
def clear_aggregator():
    """
    Clear the trace data collected in the aggregator after each tests.
    (They should be exported and popped but just in case.)
    """
    yield
    InMemoryTraceDataAggregator.get_instance()._traces.clear()


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

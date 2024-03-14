from unittest import mock

from opentelemetry import trace

from mlflow.tracing.client import get_trace_client
from mlflow.tracing.provider import _TRACER_PROVIDER_INITIALIZED, get_tracer


# Mock client getter just to count the number of calls
@mock.patch("mlflow.tracing.provider.get_trace_client", side_effect=get_trace_client)
def test_tracer_provider_singleton(mock_get_client):
    # Reset the Once object as there might be other tests that have already initialized it
    _TRACER_PROVIDER_INITIALIZED._done = False

    get_tracer("module_1")
    assert mock_get_client.call_count == 1
    assert _TRACER_PROVIDER_INITIALIZED._done is True

    tracer_provider_1 = trace.get_tracer_provider()

    get_tracer("module_2")
    assert mock_get_client.call_count == 1

    tracer_provider_2 = trace.get_tracer_provider()

    # Trace provider should be identical for different moments in time
    assert tracer_provider_1 is tracer_provider_2

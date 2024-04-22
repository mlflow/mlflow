from opentelemetry import trace

from mlflow.tracing.provider import _TRACER_PROVIDER_INITIALIZED, _get_tracer


# Mock client getter just to count the number of calls
def test_tracer_provider_singleton():
    # Reset the Once object as there might be other tests that have already initialized it
    _TRACER_PROVIDER_INITIALIZED._done = False
    _get_tracer("module_1")
    assert _TRACER_PROVIDER_INITIALIZED._done is True

    # Trace provider should be identical for different moments in time
    tracer_provider_1 = trace.get_tracer_provider()
    tracer_provider_2 = trace.get_tracer_provider()
    assert tracer_provider_1 is tracer_provider_2

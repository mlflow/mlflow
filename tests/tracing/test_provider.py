import pytest
from opentelemetry import trace

import mlflow
from mlflow.tracing.export.inference_table import InferenceTableSpanExporter
from mlflow.tracing.export.mlflow import MlflowSpanExporter
from mlflow.tracing.fluent import TRACE_BUFFER
from mlflow.tracing.processor.inference_table import InferenceTableSpanProcessor
from mlflow.tracing.processor.mlflow import MlflowSpanProcessor
from mlflow.tracing.provider import (
    _TRACER_PROVIDER_INITIALIZED,
    _get_tracer,
    _is_enabled,
    trace_disabled,
)


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


def test_span_processor_and_exporter_default():
    _TRACER_PROVIDER_INITIALIZED._done = False
    tracer = _get_tracer("test")
    processors = tracer.span_processor._span_processors
    assert len(processors) == 1
    assert isinstance(processors[0], MlflowSpanProcessor)
    assert isinstance(processors[0].span_exporter, MlflowSpanExporter)


def test_span_processor_and_exporter_model_serving(monkeypatch):
    monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", "True")

    _TRACER_PROVIDER_INITIALIZED._done = False
    tracer = _get_tracer("test")
    processors = tracer.span_processor._span_processors
    assert len(processors) == 1
    assert isinstance(processors[0], InferenceTableSpanProcessor)
    assert isinstance(processors[0].span_exporter, InferenceTableSpanExporter)


def test_disable_enable_tracing():
    @mlflow.trace
    def test_fn():
        pass

    test_fn()
    assert len(TRACE_BUFFER) == 1
    assert isinstance(_get_tracer(__name__), trace.Tracer)
    TRACE_BUFFER.clear()

    mlflow.tracing.disable()
    test_fn()
    assert len(TRACE_BUFFER) == 0
    assert isinstance(_get_tracer(__name__), trace.NoOpTracer)

    mlflow.tracing.enable()
    test_fn()
    assert len(TRACE_BUFFER) == 1
    assert isinstance(_get_tracer(__name__), trace.Tracer)
    TRACE_BUFFER.clear()


@pytest.mark.parametrize("enabled_initially", [True, False])
def test_trace_disabled_context_manager(enabled_initially):
    if not enabled_initially:
        mlflow.tracing.disable()
    assert _is_enabled() == enabled_initially

    @mlflow.trace
    def test_fn():
        pass

    with trace_disabled():
        test_fn()
        assert len(TRACE_BUFFER) == 0
        assert not _is_enabled()

    # Recover the initial state
    assert _is_enabled() == enabled_initially


def test_is_enabled():
    # Before doing anything -> tracing is considered as "on"
    assert _is_enabled()

    # Generate a trace -> tracing is still "on"
    @mlflow.trace
    def foo():
        pass

    foo()
    assert _is_enabled()

    # Disable tracing
    mlflow.tracing.disable()
    assert not _is_enabled()

    # Try to generate a trace -> tracing is still "off"
    foo()
    assert not _is_enabled()

    # Re-enable tracing
    mlflow.tracing.enable()
    assert _is_enabled()

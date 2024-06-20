from unittest import mock

import pytest
from opentelemetry import trace

import mlflow
from mlflow.tracing.export.inference_table import (
    _TRACE_BUFFER,
    InferenceTableSpanExporter,
)
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


def test_span_processor_and_exporter_model_serving(mock_databricks_serving_with_tracing_env):
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

    # enable() / disable() should not raise an exception
    with mock.patch(
        "mlflow.tracing.provider._is_enabled", side_effect=ValueError("error")
    ) as is_enabled_mock:
        mlflow.tracing.disable()
        mlflow.tracing.enable()
        assert is_enabled_mock.call_count == 2


@pytest.mark.parametrize("enabled_initially", [True, False])
def test_trace_disabled_decorator(enabled_initially):
    if not enabled_initially:
        mlflow.tracing.disable()
    assert _is_enabled() == enabled_initially

    call_count = 0

    @trace_disabled
    def test_fn():
        with mlflow.start_span(name="test_span") as span:
            span.set_attribute("key", "value")
        nonlocal call_count
        call_count += 1
        return 0

    test_fn()
    assert len(TRACE_BUFFER) == 0
    assert call_count == 1

    # Recover the initial state
    assert _is_enabled() == enabled_initially

    # Tracing should be enabled back even if the function raises an exception
    @trace_disabled
    def test_fn_raise():
        nonlocal call_count
        call_count += 1
        raise ValueError("error")

    with pytest.raises(ValueError, match="error"):
        test_fn_raise()
    assert call_count == 2

    assert len(TRACE_BUFFER) == 0
    assert _is_enabled() == enabled_initially

    # @trace_disabled should not block the decorated function even
    # if it fails to disable tracing
    with mock.patch(
        "mlflow.tracing.provider._setup_tracer_provider", side_effect=ValueError("error")
    ) as setup_mock:
        assert 0 == test_fn()
        assert call_count == 3
        assert setup_mock.call_count == (2 if enabled_initially else 0)


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

    # _is_enabled() should not raise an exception
    with mock.patch(
        "mlflow.tracing.provider.trace.get_tracer", side_effect=ValueError("error")
    ) as get_tracer_mock:
        assert not _is_enabled()
        assert get_tracer_mock.call_count == 1


@pytest.mark.parametrize("enable_mlflow_tracing", [True, False, None])
def test_enable_mlflow_tracing_switch_in_serving_fluent(monkeypatch, enable_mlflow_tracing):
    if enable_mlflow_tracing is None:
        monkeypatch.delenv("ENABLE_MLFLOW_TRACING", raising=False)
    else:
        monkeypatch.setenv("ENABLE_MLFLOW_TRACING", str(enable_mlflow_tracing).lower())
    monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", "true")

    @mlflow.trace
    def foo():
        return 1

    request_ids = ["id1", "id2", "id3"]
    with mock.patch(
        "mlflow.tracing.processor.inference_table.maybe_get_request_id", side_effect=request_ids
    ):
        for _ in range(3):
            foo()

    if enable_mlflow_tracing:
        assert sorted(_TRACE_BUFFER) == request_ids
    else:
        assert len(_TRACE_BUFFER) == 0


@pytest.mark.parametrize("enable_mlflow_tracing", [True, False])
def test_enable_mlflow_tracing_switch_in_serving_client(monkeypatch, enable_mlflow_tracing):
    monkeypatch.setenv("ENABLE_MLFLOW_TRACING", str(enable_mlflow_tracing).lower())
    monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", "true")

    client = mlflow.MlflowClient()

    def foo():
        return bar()

    @mlflow.trace
    def bar():
        return 1

    request_ids = ["123", "234"]
    with mock.patch(
        "mlflow.tracing.processor.inference_table.maybe_get_request_id", side_effect=request_ids
    ):
        client.start_trace("root")
        foo()
        if enable_mlflow_tracing:
            client.end_trace(request_id="123")

    if enable_mlflow_tracing:
        assert sorted(_TRACE_BUFFER) == request_ids
    else:
        assert len(_TRACE_BUFFER) == 0

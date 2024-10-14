from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest
from opentelemetry import trace

import mlflow
from mlflow.exceptions import MlflowTracingException
from mlflow.tracing.export.inference_table import (
    _TRACE_BUFFER,
    InferenceTableSpanExporter,
)
from mlflow.tracing.fluent import TRACE_BUFFER
from mlflow.tracing.processor.inference_table import InferenceTableSpanProcessor
from mlflow.tracing.provider import (
    _get_tracer,
    _setup_tracer_provider,
    is_tracing_enabled,
    reset_tracer_setup,
    start_span_in_context,
    trace_disabled,
)


@pytest.fixture
def mock_setup_tracer_provider():
    # To count the number of times _setup_tracer_provider is called
    with mock.patch(
        "mlflow.tracing.provider._setup_tracer_provider", side_effect=_setup_tracer_provider
    ) as setup_mock:
        yield setup_mock


def test_tracer_provider_initialized_once(mock_setup_tracer_provider):
    assert mock_setup_tracer_provider.call_count == 0
    start_span_in_context("test1")
    assert mock_setup_tracer_provider.call_count == 1

    start_span_in_context("test_2")
    start_span_in_context("test_3")
    assert mock_setup_tracer_provider.call_count == 1

    # Thread safety
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(start_span_in_context, ["test_4", "test_5"])
    assert mock_setup_tracer_provider.call_count == 1


def test_reset_tracer_setup(mock_setup_tracer_provider):
    assert mock_setup_tracer_provider.call_count == 0

    start_span_in_context("test1")
    assert mock_setup_tracer_provider.call_count == 1

    reset_tracer_setup()
    assert mock_setup_tracer_provider.call_count == 2

    start_span_in_context("test2")
    assert mock_setup_tracer_provider.call_count == 3
    assert mock_setup_tracer_provider.mock_calls == (
        [
            mock.call(),
            mock.call(disabled=True),
            mock.call(),
        ]
    )


def test_span_processor_and_exporter_model_serving(mock_databricks_serving_with_tracing_env):
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

    # enable() / disable() should only raise MlflowTracingException
    with mock.patch(
        "mlflow.tracing.provider.is_tracing_enabled", side_effect=ValueError("error")
    ) as is_enabled_mock:
        with pytest.raises(MlflowTracingException, match="error"):
            mlflow.tracing.disable()
        assert is_enabled_mock.call_count == 1

        with pytest.raises(MlflowTracingException, match="error"):
            mlflow.tracing.enable()
        assert is_enabled_mock.call_count == 2


@pytest.mark.parametrize("enabled_initially", [True, False])
def test_trace_disabled_decorator(enabled_initially):
    if not enabled_initially:
        mlflow.tracing.disable()
    assert is_tracing_enabled() == enabled_initially
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
    assert is_tracing_enabled() == enabled_initially

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
    assert is_tracing_enabled() == enabled_initially

    # @trace_disabled should not block the decorated function even
    # if it fails to disable tracing
    with mock.patch(
        "mlflow.tracing.provider.disable", side_effect=MlflowTracingException("error")
    ) as disable_mock:
        assert test_fn() == 0
        assert call_count == 3
        assert disable_mock.call_count == (1 if enabled_initially else 0)

    with mock.patch(
        "mlflow.tracing.provider.enable", side_effect=MlflowTracingException("error")
    ) as enable_mock:
        assert test_fn() == 0
        assert call_count == 4
        assert enable_mock.call_count == (1 if enabled_initially else 0)


def test_disable_enable_tracing_not_mutate_otel_provider():
    # This test validates that disable/enable MLflow tracing does not mutate the OpenTelemetry's
    # global tracer provider instance.
    otel_tracer_provider = trace.get_tracer_provider()

    mlflow.tracing.disable()
    assert trace.get_tracer_provider() is otel_tracer_provider

    mlflow.tracing.enable()
    assert trace.get_tracer_provider() is otel_tracer_provider

    @trace_disabled
    def test_fn():
        assert trace.get_tracer_provider() is otel_tracer_provider

    test_fn()
    assert trace.get_tracer_provider() is otel_tracer_provider


def test_is_tracing_enabled():
    # Before doing anything -> tracing is considered as "on"
    assert is_tracing_enabled()

    # Generate a trace -> tracing is still "on"
    @mlflow.trace
    def foo():
        pass

    foo()
    assert is_tracing_enabled()

    # Disable tracing
    mlflow.tracing.disable()
    assert is_tracing_enabled() is False

    # Try to generate a trace -> tracing is still "off"
    foo()
    assert is_tracing_enabled() is False

    # Re-enable tracing
    mlflow.tracing.enable()
    assert is_tracing_enabled() is True

    # is_tracing_enabled() should only raise MlflowTracingException
    with mock.patch(
        "mlflow.tracing.provider._get_tracer", side_effect=ValueError("error")
    ) as get_tracer_mock:
        with pytest.raises(MlflowTracingException, match="error"):
            assert is_tracing_enabled() is False
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

from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest
from opentelemetry import trace

import mlflow
import mlflow.tracking._tracking_service
from mlflow.environment_variables import (
    MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT,
    MLFLOW_TRACE_SAMPLING_RATIO,
)
from mlflow.exceptions import MlflowTracingException
from mlflow.tracing.destination import Databricks, MlflowExperiment
from mlflow.tracing.export.inference_table import (
    _TRACE_BUFFER,
    InferenceTableSpanExporter,
)
from mlflow.tracing.export.mlflow_v3 import MlflowV3SpanExporter
from mlflow.tracing.fluent import start_span_no_context
from mlflow.tracing.processor.inference_table import InferenceTableSpanProcessor
from mlflow.tracing.processor.mlflow_v3 import MlflowV3SpanProcessor
from mlflow.tracing.processor.otel import OtelSpanProcessor
from mlflow.tracing.provider import (
    _get_tracer,
    _setup_tracer_provider,
    is_tracing_enabled,
    start_span_in_context,
    trace_disabled,
)

from tests.tracing.helper import get_traces, purge_traces, skip_when_testing_trace_sdk


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

    mlflow.tracing.reset()
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


def test_set_destination_mlflow_experiment(monkeypatch):
    # Set destination with experiment_id
    mlflow.tracing.set_destination(destination=MlflowExperiment(experiment_id="123"))

    tracer = _get_tracer("test")
    processors = tracer.span_processor._span_processors
    assert len(processors) == 1
    assert isinstance(processors[0], MlflowV3SpanProcessor)
    assert isinstance(processors[0].span_exporter, MlflowV3SpanExporter)

    # Set destination with experiment_id and tracking_uri
    mlflow.tracing.set_destination(destination=MlflowExperiment(experiment_id="456"))

    tracer = _get_tracer("test")
    processors = tracer.span_processor._span_processors

    # Experiment with Databricks tracking URI -> V3 exporter should be used
    mlflow.tracing.set_destination(destination=MlflowExperiment(experiment_id="456"))

    tracer = _get_tracer("test")
    processors = tracer.span_processor._span_processors
    assert isinstance(processors[0], MlflowV3SpanProcessor)
    assert isinstance(processors[0].span_exporter, MlflowV3SpanExporter)


def test_set_destination_databricks(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "databricks")
    mlflow.tracing.set_destination(destination=Databricks(experiment_id="123"))

    tracer = _get_tracer("test")
    processors = tracer.span_processor._span_processors
    assert len(processors) == 1
    assert isinstance(processors[0], MlflowV3SpanProcessor)
    assert isinstance(processors[0].span_exporter, MlflowV3SpanExporter)


def test_set_destination_databricks_serving(mock_databricks_serving_with_tracing_env, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "databricks")
    mlflow.tracing.set_destination(destination=Databricks(experiment_id="123"))

    tracer = _get_tracer("test")
    processors = tracer.span_processor._span_processors
    assert len(processors) == 1
    assert isinstance(processors[0], MlflowV3SpanProcessor)
    assert isinstance(processors[0].span_exporter, MlflowV3SpanExporter)


def test_disable_enable_tracing():
    @mlflow.trace
    def test_fn():
        pass

    test_fn()
    assert len(get_traces()) == 1
    assert isinstance(_get_tracer(__name__), trace.Tracer)
    purge_traces()

    mlflow.tracing.disable()
    test_fn()
    assert len(get_traces()) == 0
    assert isinstance(_get_tracer(__name__), trace.NoOpTracer)

    mlflow.tracing.enable()
    test_fn()
    assert len(get_traces()) == 1
    assert isinstance(_get_tracer(__name__), trace.Tracer)

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
    assert len(get_traces()) == 0
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

    assert len(get_traces()) == 0
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

    def foo():
        return bar()

    @mlflow.trace
    def bar():
        return 1

    request_ids = ["123", "234"]
    with mock.patch(
        "mlflow.tracing.processor.inference_table.maybe_get_request_id", side_effect=request_ids
    ):
        span = start_span_no_context("root")
        foo()
        if enable_mlflow_tracing:
            span.end()

    if enable_mlflow_tracing:
        assert sorted(_TRACE_BUFFER) == request_ids
    else:
        assert len(_TRACE_BUFFER) == 0


def test_sampling_ratio(monkeypatch):
    @mlflow.trace
    def test_function():
        return "test"

    # Test with 100% sampling (default)
    for _ in range(10):
        test_function()

    traces = get_traces()
    assert len(traces) == 10
    purge_traces()

    # Test with 0% sampling
    monkeypatch.setenv(MLFLOW_TRACE_SAMPLING_RATIO.name, "0.0")
    mlflow.tracing.reset()

    for _ in range(10):
        test_function()

    traces = get_traces()
    assert len(traces) == 0
    purge_traces()

    # With 50% sampling and 100 runs, we expect around 50 traces
    # but due to randomness, we check for a reasonable range
    monkeypatch.setenv(MLFLOW_TRACE_SAMPLING_RATIO.name, "0.5")
    mlflow.tracing.reset()

    for _ in range(100):
        test_function()

    traces = get_traces()
    assert 30 <= len(traces) <= 70, (
        f"Expected around 50 traces with 0.5 sampling, got {len(traces)}"
    )


def test_otlp_exclusive_vs_dual_export(monkeypatch):
    """Test OTLP exclusive mode vs dual export mode."""
    from mlflow.environment_variables import MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT
    from mlflow.tracing.processor.otel import OtelSpanProcessor
    from mlflow.tracing.provider import _get_tracer

    # Test 1: OTLP exclusive mode (dual export = false, default)
    monkeypatch.setenv(MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT.name, "false")
    with mock.patch("mlflow.tracing.provider.should_use_otlp_exporter", return_value=True):
        with mock.patch("mlflow.tracing.provider.get_otlp_exporter") as mock_get_exporter:
            mock_get_exporter.return_value = mock.MagicMock()

            mlflow.tracing.reset()
            tracer = _get_tracer("test")

            from mlflow.tracing.provider import _MLFLOW_TRACER_PROVIDER

            assert _MLFLOW_TRACER_PROVIDER is not None
            processors = tracer.span_processor._span_processors

            # Should have only OTLP processor as primary
            assert len(processors) == 1
            assert isinstance(processors[0], OtelSpanProcessor)

    # Test 2: Dual export mode (both MLflow and OTLP)
    monkeypatch.setenv(MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT.name, "true")
    with mock.patch("mlflow.tracing.provider.should_use_otlp_exporter", return_value=True):
        with mock.patch("mlflow.tracing.provider.get_otlp_exporter") as mock_get_exporter:
            mock_get_exporter.return_value = mock.MagicMock()

            mlflow.tracing.reset()
            tracer = _get_tracer("test")

            from mlflow.tracing.provider import _MLFLOW_TRACER_PROVIDER

            assert _MLFLOW_TRACER_PROVIDER is not None
            processors = tracer.span_processor._span_processors

            # Should have both processors
            assert len(processors) == 2
            assert isinstance(processors[0], OtelSpanProcessor)
            assert isinstance(processors[1], MlflowV3SpanProcessor)


@skip_when_testing_trace_sdk
@pytest.mark.parametrize("dual_export", [False, True])
def test_metrics_export_with_otlp_trace_export(monkeypatch, dual_export):
    """Test metrics export configuration when OTLP is enabled."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4317")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://localhost:9090")

    if dual_export:
        monkeypatch.setenv(MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT.name, "true")

    mlflow.tracing.reset()
    tracer = _get_tracer("test")

    if dual_export:
        processors = tracer.span_processor._span_processors
        assert len(processors) == 2
        assert isinstance(processors[0], OtelSpanProcessor)
        assert isinstance(processors[1], MlflowV3SpanProcessor)

        # In dual export, MLflow processor exports metrics, OTLP doesn't
        assert processors[0]._export_metrics is False
        assert processors[1]._export_metrics is True
    else:
        processors = tracer.span_processor._span_processors
        assert len(processors) == 1
        assert isinstance(processors[0], OtelSpanProcessor)
        assert processors[0]._export_metrics is True


@skip_when_testing_trace_sdk
def test_metrics_export_without_otlp_trace_export(monkeypatch):
    """Test metrics export configuration when OTLP is disabled."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://localhost:9090")

    # No OTLP tracing endpoints set
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    mlflow.tracing.reset()
    tracer = _get_tracer("test")

    processors = tracer.span_processor._span_processors
    assert len(processors) == 1
    assert isinstance(processors[0], MlflowV3SpanProcessor)
    assert processors[0]._export_metrics is True

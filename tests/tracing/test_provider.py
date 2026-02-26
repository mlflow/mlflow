from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest
from opentelemetry import trace

import mlflow
from mlflow.entities.trace_location import (
    MlflowExperimentLocation,
    UCSchemaLocation,
    UnityCatalog,
)
from mlflow.environment_variables import (
    MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT,
    MLFLOW_TRACE_SAMPLING_RATIO,
    MLFLOW_USE_DEFAULT_TRACER_PROVIDER,
)
from mlflow.exceptions import MlflowTracingException
from mlflow.tracing.destination import Databricks, MlflowExperiment
from mlflow.tracing.export.inference_table import (
    _TRACE_BUFFER,
    InferenceTableSpanExporter,
)
from mlflow.tracing.export.mlflow_v3 import MlflowV3SpanExporter
from mlflow.tracing.export.uc_table import DatabricksUCTableSpanExporter
from mlflow.tracing.fluent import start_span_no_context
from mlflow.tracing.processor.inference_table import InferenceTableSpanProcessor
from mlflow.tracing.processor.mlflow_v3 import MlflowV3SpanProcessor
from mlflow.tracing.processor.otel import OtelSpanProcessor
from mlflow.tracing.processor.uc_table import DatabricksUCTableSpanProcessor
from mlflow.tracing.provider import (
    _get_tracer,
    _initialize_tracer_provider,
    is_tracing_enabled,
    start_span_in_context,
    trace_disabled,
)
from mlflow.tracing.utils import get_active_spans_table_name

from tests.tracing.helper import get_traces, purge_traces, skip_when_testing_trace_sdk


@pytest.fixture
def mock_setup_tracer_provider():
    # To count the number of times _initialize_tracer_provider is called
    with mock.patch(
        "mlflow.tracing.provider._initialize_tracer_provider",
        side_effect=_initialize_tracer_provider,
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
    mlflow.tracing.set_destination(destination=MlflowExperimentLocation(experiment_id="123"))

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


def test_set_destination_databricks_uc(monkeypatch):
    mlflow.tracing.set_destination(
        destination=UCSchemaLocation(
            catalog_name="catalog",
            schema_name="schema",
        )
    )

    tracer = _get_tracer("test")
    processors = tracer.span_processor._span_processors
    assert len(processors) == 1
    assert isinstance(processors[0], DatabricksUCTableSpanProcessor)
    assert isinstance(processors[0].span_exporter, DatabricksUCTableSpanExporter)
    assert get_active_spans_table_name() == "catalog.schema.mlflow_experiment_trace_otel_spans"


def test_set_destination_databricks_uc_with_oltp_env_no_dual_export(monkeypatch):
    # set_destination is called but OLTP is also set w/o dual export mode enabled
    monkeypatch.setenv(MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT.name, "false")
    with (
        mock.patch("mlflow.tracing.provider.should_use_otlp_exporter", return_value=True),
        mock.patch("mlflow.tracing.provider.get_otlp_exporter") as mock_get_exporter,
    ):
        mock_get_exporter.return_value = mock.MagicMock()

        mlflow.tracing.reset()
        mlflow.tracing.set_destination(
            destination=UCSchemaLocation(
                catalog_name="catalog",
                schema_name="schema",
            )
        )
        tracer = _get_tracer("test")
        processors = tracer.span_processor._span_processors
        assert len(processors) == 1
        assert isinstance(processors[0], DatabricksUCTableSpanProcessor)
        assert isinstance(processors[0].span_exporter, DatabricksUCTableSpanExporter)
        assert get_active_spans_table_name() == "catalog.schema.mlflow_experiment_trace_otel_spans"


def test_set_destination_databricks_uc_with_oltp_env_with_dual_export(monkeypatch):
    # set_destination is called but OLTP is also set w/ dual export mode enabled
    monkeypatch.setenv(MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT.name, "true")
    with (
        mock.patch("mlflow.tracing.provider.should_use_otlp_exporter", return_value=True),
        mock.patch("mlflow.tracing.provider.get_otlp_exporter") as mock_get_exporter,
    ):
        mock_get_exporter.return_value = mock.MagicMock()

        mlflow.tracing.reset()
        mlflow.tracing.set_destination(
            destination=UCSchemaLocation(
                catalog_name="catalog",
                schema_name="schema",
            )
        )
        tracer = _get_tracer("test")
        processors = tracer.span_processor._span_processors
        assert len(processors) == 2
        assert isinstance(processors[0], DatabricksUCTableSpanProcessor)
        assert isinstance(processors[0].span_exporter, DatabricksUCTableSpanExporter)
        # OTLP processor needs to be there for dual export mode
        assert isinstance(processors[1], OtelSpanProcessor)
        assert get_active_spans_table_name() == "catalog.schema.mlflow_experiment_trace_otel_spans"


def test_set_destination_from_env_var_mlflow_experiment(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACING_DESTINATION", "123")

    tracer = _get_tracer("test")
    processors = tracer.span_processor._span_processors
    assert len(processors) == 1
    assert isinstance(processors[0], MlflowV3SpanProcessor)
    assert isinstance(processors[0].span_exporter, MlflowV3SpanExporter)


def test_set_destination_from_env_var_databricks_uc(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACING_DESTINATION", "catalog.schema")

    tracer = _get_tracer("test")
    processors = tracer.span_processor._span_processors
    assert len(processors) == 1
    assert isinstance(processors[0], DatabricksUCTableSpanProcessor)
    assert isinstance(processors[0].span_exporter, DatabricksUCTableSpanExporter)
    assert get_active_spans_table_name() == "catalog.schema.mlflow_experiment_trace_otel_spans"


def test_set_destination_from_env_var_databricks_uc_with_table_prefix(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACING_DESTINATION", "catalog.schema.prefix")

    from mlflow.tracing.provider import _MLFLOW_TRACE_USER_DESTINATION

    destination = _MLFLOW_TRACE_USER_DESTINATION.get()
    assert isinstance(destination, UnityCatalog)
    assert destination.catalog_name == "catalog"
    assert destination.schema_name == "schema"
    assert destination.table_prefix == "prefix"


def test_destination_resolution_precedence_with_experiment_derived(monkeypatch):
    from mlflow.tracing.provider import _MLFLOW_TRACE_USER_DESTINATION

    _MLFLOW_TRACE_USER_DESTINATION.reset()
    monkeypatch.setenv("MLFLOW_TRACING_DESTINATION", "catalog.schema.env")
    monkeypatch.setattr("mlflow.tracking.fluent._get_experiment_id", lambda: "exp-1")

    experiment_derived = UnityCatalog("catalog", "schema", table_prefix="exp")
    global_destination = UnityCatalog("catalog", "schema", table_prefix="global")
    local_destination = UnityCatalog("catalog", "schema", table_prefix="local")

    _MLFLOW_TRACE_USER_DESTINATION.set_experiment_derived(
        experiment_derived,
        experiment_id="exp-1",
    )
    assert _MLFLOW_TRACE_USER_DESTINATION.get().table_prefix == "exp"

    _MLFLOW_TRACE_USER_DESTINATION.set(global_destination)
    assert _MLFLOW_TRACE_USER_DESTINATION.get().table_prefix == "global"

    _MLFLOW_TRACE_USER_DESTINATION.set(local_destination, context_local=True)
    assert _MLFLOW_TRACE_USER_DESTINATION.get().table_prefix == "local"
    _MLFLOW_TRACE_USER_DESTINATION.reset()


def test_experiment_derived_destination_invalidates_when_experiment_changes(monkeypatch):
    from mlflow.tracing.provider import _MLFLOW_TRACE_USER_DESTINATION

    _MLFLOW_TRACE_USER_DESTINATION.reset()
    monkeypatch.setenv("MLFLOW_TRACING_DESTINATION", "catalog.schema.env")
    monkeypatch.setattr("mlflow.tracking.fluent._get_experiment_id", lambda: "exp-2")

    experiment_derived = UnityCatalog("catalog", "schema", table_prefix="exp")
    _MLFLOW_TRACE_USER_DESTINATION.set_experiment_derived(
        experiment_derived,
        experiment_id="exp-1",
    )

    # Cached experiment-derived value no longer matches active experiment.
    # Registry should invalidate it and fall through to env resolution.
    destination = _MLFLOW_TRACE_USER_DESTINATION.get()
    assert isinstance(destination, UnityCatalog)
    assert destination.table_prefix == "env"
    assert _MLFLOW_TRACE_USER_DESTINATION._experiment_derived_value is None


def test_set_destination_in_model_serving(mock_databricks_serving_with_tracing_env, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "databricks")
    monkeypatch.setenv("MLFLOW_TRACING_DESTINATION", "catalog.schema")

    tracer = _get_tracer("test")
    processors = tracer.span_processor._span_processors
    assert len(processors) == 1
    assert isinstance(processors[0], DatabricksUCTableSpanProcessor)
    assert isinstance(processors[0].span_exporter, DatabricksUCTableSpanExporter)
    assert get_active_spans_table_name() == "catalog.schema.mlflow_experiment_trace_otel_spans"


def test_set_destination_deprecated_classes():
    from mlflow.tracing.provider import _MLFLOW_TRACE_USER_DESTINATION

    with pytest.warns(FutureWarning, match="`mlflow.tracing.destination.MlflowExperiment``"):
        mlflow.tracing.set_destination(destination=MlflowExperiment(experiment_id="123"))

    destination = _MLFLOW_TRACE_USER_DESTINATION.get()
    assert isinstance(destination, MlflowExperimentLocation)
    assert destination.experiment_id == "123"

    with pytest.warns(FutureWarning, match="`mlflow.tracing.destination.Databricks`"):
        mlflow.tracing.set_destination(destination=Databricks(experiment_id="123"))

    destination = _MLFLOW_TRACE_USER_DESTINATION.get()
    assert isinstance(destination, MlflowExperimentLocation)
    assert destination.experiment_id == "123"


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


def test_disable_enable_tracing_not_mutate_otel_provider(monkeypatch):
    monkeypatch.setenv(MLFLOW_USE_DEFAULT_TRACER_PROVIDER.name, "true")

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


def test_otlp_exclusive_vs_dual_export_with_no_set_location(monkeypatch):
    from mlflow.environment_variables import MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT
    from mlflow.tracing.processor.otel import OtelSpanProcessor
    from mlflow.tracing.provider import _get_tracer

    # Test 1: OTLP exclusive mode (dual export = false, default)
    monkeypatch.setenv(MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT.name, "false")
    with (
        mock.patch("mlflow.tracing.provider.should_use_otlp_exporter", return_value=True),
        mock.patch("mlflow.tracing.provider.get_otlp_exporter") as mock_get_exporter,
    ):
        mock_get_exporter.return_value = mock.MagicMock()

        mlflow.tracing.reset()
        tracer = _get_tracer("test")

        processors = tracer.span_processor._span_processors

        # Should have only OTLP processor as primary
        assert len(processors) == 1
        assert isinstance(processors[0], OtelSpanProcessor)

    # Test 2: Dual export mode (both MLflow and OTLP)
    monkeypatch.setenv(MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT.name, "true")
    with (
        mock.patch("mlflow.tracing.provider.should_use_otlp_exporter", return_value=True),
        mock.patch("mlflow.tracing.provider.get_otlp_exporter") as mock_get_exporter,
    ):
        mock_get_exporter.return_value = mock.MagicMock()

        mlflow.tracing.reset()
        tracer = _get_tracer("test")

        processors = tracer.span_processor._span_processors

        # Should have both processors
        assert len(processors) == 2
        assert isinstance(processors[0], OtelSpanProcessor)
        assert isinstance(processors[1], MlflowV3SpanProcessor)


@skip_when_testing_trace_sdk
@pytest.mark.parametrize("dual_export", [False, True])
def test_metrics_export_with_otlp_trace_export(monkeypatch, dual_export):
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


def test_otel_resource_attributes(monkeypatch):
    tracer = _get_tracer("test")
    # By default, only MLflow's SDK attributes are set on an empty resource
    assert tracer.resource.attributes == {
        "telemetry.sdk.language": "python",
        "telemetry.sdk.name": "mlflow",
        "telemetry.sdk.version": mlflow.__version__,
    }

    mlflow.tracing.reset()
    # When otel attributes are set explicitly, MLflow's SDK attributes are not set on the resource
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "favorite.fruit=apple,color=red")
    tracer = _get_tracer("test")
    assert dict(tracer.resource.attributes) == {
        "favorite.fruit": "apple",
        "color": "red",
        "telemetry.sdk.language": "python",
        "telemetry.sdk.name": "mlflow",
        "telemetry.sdk.version": mlflow.__version__,
        "service.name": "unknown_service",
    }

    # Service name should be propagated from the env var
    mlflow.tracing.reset()
    monkeypatch.setenv("OTEL_SERVICE_NAME", "test-service")
    monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)
    tracer = _get_tracer("test")
    assert dict(tracer.resource.attributes) == {
        "service.name": "test-service",
        "telemetry.sdk.language": "python",
        "telemetry.sdk.name": "mlflow",
        "telemetry.sdk.version": mlflow.__version__,
    }

    # Invalid env var should be ignored and does not block the tracer provider initialization
    mlflow.tracing.reset()
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "invalid")
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    tracer = _get_tracer("test")
    assert dict(tracer.resource.attributes) == {
        "service.name": "unknown_service",
        "telemetry.sdk.language": "python",
        "telemetry.sdk.name": "mlflow",
        "telemetry.sdk.version": mlflow.__version__,
    }


_RESOLVE_UC_DEST_TAG = "mlflow.experiment.databricksTelemetryDestinationId"


def _make_experiment(tags=None):
    from mlflow.entities import Experiment
    from mlflow.entities.experiment_tag import ExperimentTag

    tag_entities = [ExperimentTag(k, v) for k, v in (tags or {}).items()] if tags else []
    return Experiment(
        experiment_id="123",
        name="test",
        artifact_location="file:/tmp",
        lifecycle_stage="active",
        tags=tag_entities,
    )


def test_resolve_uc_location_from_experiment_tag():
    from mlflow.tracing.provider import _resolve_experiment_uc_location

    mlflow.tracing.reset()
    resolved = UnityCatalog("cat", "sch", table_prefix="pfx")

    with (
        mock.patch(
            "mlflow.tracing.provider.mlflow.get_tracking_uri",
            return_value="databricks",
        ),
        mock.patch("mlflow.tracking.fluent._get_experiment_id", return_value="123"),
        mock.patch("mlflow.tracking._tracking_service.utils._get_store") as mock_store_fn,
        mock.patch("mlflow.tracing.client.TracingClient") as tc_cls,
    ):
        mock_store_fn.return_value.get_experiment.return_value = _make_experiment(
            tags={_RESOLVE_UC_DEST_TAG: "some-uuid"}
        )
        tc_cls.return_value._get_trace_location.return_value = resolved

        result = _resolve_experiment_uc_location()

        assert result == resolved
        tc_cls.return_value._get_trace_location.assert_called_once_with("some-uuid")

    mlflow.tracing.reset()


def test_resolve_uc_location_returns_none_when_no_tag():
    from mlflow.tracing.provider import _resolve_experiment_uc_location

    mlflow.tracing.reset()

    with (
        mock.patch(
            "mlflow.tracing.provider.mlflow.get_tracking_uri",
            return_value="databricks",
        ),
        mock.patch("mlflow.tracking.fluent._get_experiment_id", return_value="123"),
        mock.patch("mlflow.tracking._tracking_service.utils._get_store") as mock_store_fn,
    ):
        mock_store_fn.return_value.get_experiment.return_value = _make_experiment()

        assert _resolve_experiment_uc_location() is None

    mlflow.tracing.reset()


def test_resolve_uc_location_returns_none_for_non_databricks():
    from mlflow.tracing.provider import _resolve_experiment_uc_location

    mlflow.tracing.reset()

    with mock.patch(
        "mlflow.tracing.provider.mlflow.get_tracking_uri",
        return_value="http://local",
    ):
        assert _resolve_experiment_uc_location() is None

    mlflow.tracing.reset()


def test_resolve_uc_location_returns_none_when_lookup_fails():
    from mlflow.tracing.provider import _resolve_experiment_uc_location

    mlflow.tracing.reset()

    with (
        mock.patch(
            "mlflow.tracing.provider.mlflow.get_tracking_uri",
            return_value="databricks",
        ),
        mock.patch("mlflow.tracking.fluent._get_experiment_id", return_value="123"),
        mock.patch("mlflow.tracking._tracking_service.utils._get_store") as mock_store_fn,
        mock.patch("mlflow.tracing.client.TracingClient") as tc_cls,
    ):
        mock_store_fn.return_value.get_experiment.return_value = _make_experiment(
            tags={_RESOLVE_UC_DEST_TAG: "some-uuid"}
        )
        tc_cls.return_value._get_trace_location.side_effect = RuntimeError("boom")

        assert _resolve_experiment_uc_location() is None

    mlflow.tracing.reset()

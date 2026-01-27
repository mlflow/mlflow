from unittest import mock

import pytest

import mlflow.tracking.context.default_context
from mlflow.entities.span import LiveSpan
from mlflow.entities.telemetry_profile import (
    Exporter,
    ExporterType,
    TelemetryProfile,
    UnityCatalogTablesConfig,
)
from mlflow.entities.trace_location import TraceLocationType
from mlflow.entities.trace_state import TraceState
from mlflow.environment_variables import MLFLOW_TRACKING_USERNAME
from mlflow.exceptions import MlflowException
from mlflow.tracing.processor.uc_table_with_otel import DatabricksUCTableWithOtelSpanProcessor
from mlflow.tracing.trace_manager import InMemoryTraceManager

from tests.tracing.helper import (
    create_mock_otel_span,
    create_test_trace_info,
)


def _create_test_telemetry_profile(
    catalog: str = "test_catalog",
    schema: str = "test_schema",
    table_prefix: str = "prefix_",
) -> TelemetryProfile:
    config = UnityCatalogTablesConfig(
        uc_catalog=catalog,
        uc_schema=schema,
        uc_table_prefix=table_prefix,
    )
    exporter = Exporter(type=ExporterType.UNITY_CATALOG_TABLES, uc_tables=config)
    return TelemetryProfile(
        profile_id="test-profile-123",
        profile_name="Test Profile",
        exporters=[exporter],
    )


def test_on_start_creates_trace_with_uc_table_prefix_location(monkeypatch):
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "alice")

    trace_id = 12345
    span = create_mock_otel_span(trace_id=trace_id, span_id=1, parent_id=None, start_time=5_000_000)

    telemetry_profile = _create_test_telemetry_profile()
    mock_exporter = mock.MagicMock()

    processor = DatabricksUCTableWithOtelSpanProcessor(
        span_exporter=mock_exporter,
        telemetry_profile=telemetry_profile,
    )
    processor.on_start(span)

    # Check that trace was created in trace manager
    trace_manager = InMemoryTraceManager.get_instance()
    traces = trace_manager._traces
    assert len(traces) == 1

    # Get the created trace
    created_trace = list(traces.values())[0]
    trace_info = created_trace.info

    # Verify trace location is UC_TABLE_PREFIX type
    assert trace_info.trace_location.type == TraceLocationType.UC_TABLE_PREFIX
    uc_table_prefix = trace_info.trace_location.uc_table_prefix
    assert uc_table_prefix.catalog_name == "test_catalog"
    assert uc_table_prefix.schema_name == "test_schema"
    assert uc_table_prefix.table_prefix == "prefix_"

    # Verify trace state and timing
    assert trace_info.state == TraceState.IN_PROGRESS
    assert trace_info.request_time == 5  # 5_000_000 nanoseconds -> 5 milliseconds
    assert trace_info.execution_duration is None


def test_on_start_missing_uc_tables_config(monkeypatch):
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "alice")

    trace_id = 12345
    span = create_mock_otel_span(trace_id=trace_id, span_id=1, parent_id=None, start_time=5_000_000)

    # Create profile without UC tables config
    profile = TelemetryProfile(exporters=[])
    mock_exporter = mock.MagicMock()

    processor = DatabricksUCTableWithOtelSpanProcessor(
        span_exporter=mock_exporter,
        telemetry_profile=profile,
    )

    with pytest.raises(MlflowException, match="does not contain a UnityCatalogTablesConfig"):
        processor.on_start(span)


def test_on_start_missing_catalog_or_schema(monkeypatch):
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "alice")

    trace_id = 12345
    span = create_mock_otel_span(trace_id=trace_id, span_id=1, parent_id=None, start_time=5_000_000)

    # Create profile with missing catalog
    config = UnityCatalogTablesConfig(uc_catalog=None, uc_schema="test_schema")
    exporter = Exporter(type=ExporterType.UNITY_CATALOG_TABLES, uc_tables=config)
    profile = TelemetryProfile(exporters=[exporter])
    mock_exporter = mock.MagicMock()

    processor = DatabricksUCTableWithOtelSpanProcessor(
        span_exporter=mock_exporter,
        telemetry_profile=profile,
    )

    with pytest.raises(MlflowException, match="missing uc_catalog or uc_schema"):
        processor.on_start(span)


def test_on_start_child_span_uses_existing_trace(monkeypatch):
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "alice")

    trace_id = 12345
    telemetry_profile = _create_test_telemetry_profile()
    mock_exporter = mock.MagicMock()

    processor = DatabricksUCTableWithOtelSpanProcessor(
        span_exporter=mock_exporter,
        telemetry_profile=telemetry_profile,
    )

    # First, create root span
    root_span = create_mock_otel_span(
        trace_id=trace_id, span_id=1, parent_id=None, start_time=5_000_000
    )
    processor.on_start(root_span)

    # Get the trace ID from trace manager
    trace_manager = InMemoryTraceManager.get_instance()
    mlflow_trace_id = trace_manager.get_mlflow_trace_id_from_otel_id(trace_id)
    assert mlflow_trace_id is not None

    # Now create child span
    child_span = create_mock_otel_span(
        trace_id=trace_id, span_id=2, parent_id=1, start_time=6_000_000
    )
    processor.on_start(child_span)

    # Should still only have one trace
    assert len(trace_manager._traces) == 1


def test_trace_id_generation_uses_catalog_schema():
    trace_id = 12345
    span = create_mock_otel_span(trace_id=trace_id, span_id=1, parent_id=None, start_time=5_000_000)

    telemetry_profile = _create_test_telemetry_profile(
        catalog="my_catalog", schema="my_schema", table_prefix="pre_"
    )

    with mock.patch(
        "mlflow.tracing.processor.uc_table_with_otel.generate_trace_id_v4",
        return_value="trace:/my_catalog.my_schema/12345",
    ) as mock_generate_trace_id:
        processor = DatabricksUCTableWithOtelSpanProcessor(
            span_exporter=mock.MagicMock(),
            telemetry_profile=telemetry_profile,
        )
        processor.on_start(span)

        # Verify generate_trace_id_v4 was called with catalog.schema as location_id
        mock_generate_trace_id.assert_called_once_with(span, "my_catalog.my_schema")


def test_trace_metadata_has_schema_version_4():
    trace_id = 12345
    span = create_mock_otel_span(trace_id=trace_id, span_id=1, parent_id=None, start_time=5_000_000)

    telemetry_profile = _create_test_telemetry_profile()
    processor = DatabricksUCTableWithOtelSpanProcessor(
        span_exporter=mock.MagicMock(),
        telemetry_profile=telemetry_profile,
    )
    processor.on_start(span)

    # Get the created trace
    trace_manager = InMemoryTraceManager.get_instance()
    created_trace = list(trace_manager._traces.values())[0]
    trace_info = created_trace.info

    # Verify schema version is 4
    assert trace_info.trace_metadata.get("mlflow.trace_schema.version") == "4"


def test_on_end_calls_batch_span_processor():
    trace_info = create_test_trace_info("request_id", 0)
    trace_manager = InMemoryTraceManager.get_instance()
    trace_manager.register_trace("trace_id", trace_info)

    otel_span = create_mock_otel_span(
        name="foo",
        trace_id="trace_id",
        span_id=1,
        parent_id=None,
        start_time=5_000_000,
        end_time=9_000_000,
    )
    span = LiveSpan(otel_span, "request_id")
    span.set_status("OK")

    telemetry_profile = _create_test_telemetry_profile()
    mock_exporter = mock.MagicMock()

    processor = DatabricksUCTableWithOtelSpanProcessor(
        span_exporter=mock_exporter,
        telemetry_profile=telemetry_profile,
    )

    # Mock the parent class on_end to verify it's called
    with mock.patch.object(
        processor.__class__.__bases__[1], "on_end", autospec=True
    ) as mock_batch_on_end:
        processor.on_end(otel_span)
        mock_batch_on_end.assert_called_once()


def test_constructor_stores_telemetry_profile():
    telemetry_profile = _create_test_telemetry_profile()
    mock_exporter = mock.MagicMock()

    processor = DatabricksUCTableWithOtelSpanProcessor(
        span_exporter=mock_exporter,
        telemetry_profile=telemetry_profile,
    )

    assert processor._telemetry_profile == telemetry_profile
    assert processor._export_metrics is False


def test_constructor_with_export_metrics():
    telemetry_profile = _create_test_telemetry_profile()
    mock_exporter = mock.MagicMock()

    processor = DatabricksUCTableWithOtelSpanProcessor(
        span_exporter=mock_exporter,
        telemetry_profile=telemetry_profile,
        export_metrics=True,
    )

    assert processor._export_metrics is True

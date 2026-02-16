from unittest import mock

import pytest

import mlflow.tracking.context.default_context
from mlflow.entities.span import LiveSpan
from mlflow.entities.trace_location import TraceLocationType
from mlflow.entities.trace_state import TraceState
from mlflow.environment_variables import MLFLOW_TRACKING_USERNAME
from mlflow.exceptions import MlflowException
from mlflow.tracing.processor.uc_table import DatabricksUCTableSpanProcessor
from mlflow.tracing.trace_manager import InMemoryTraceManager

from tests.tracing.helper import (
    create_mock_otel_span,
    create_test_trace_info,
)


def test_on_start_with_uc_table_name(monkeypatch):
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "alice")

    # Root span should create a new trace on start
    trace_id = 12345
    span = create_mock_otel_span(trace_id=trace_id, span_id=1, parent_id=None, start_time=5_000_000)

    # Mock get_active_spans_table_name to return a UC table name
    with mock.patch(
        "mlflow.tracing.processor.uc_table.get_active_spans_table_name",
        return_value="catalog1.schema1.spans_table",
    ):
        processor = DatabricksUCTableSpanProcessor(span_exporter=mock.MagicMock())
        processor.on_start(span)

    # Check that trace was created in trace manager
    trace_manager = InMemoryTraceManager.get_instance()
    traces = trace_manager._traces
    assert len(traces) == 1

    # Get the created trace
    created_trace = list(traces.values())[0]
    trace_info = created_trace.info

    # Verify trace location is UC_SCHEMA type
    assert trace_info.trace_location.type == TraceLocationType.UC_SCHEMA
    uc_schema = trace_info.trace_location.uc_schema
    assert uc_schema.catalog_name == "catalog1"
    assert uc_schema.schema_name == "schema1"

    # Verify trace state and timing
    assert trace_info.state == TraceState.IN_PROGRESS
    assert trace_info.request_time == 5  # 5_000_000 nanoseconds -> 5 milliseconds
    assert trace_info.execution_duration is None


def test_on_start_without_uc_table_name(monkeypatch):
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "alice")

    # Root span should create a new trace on start
    trace_id = 12345
    span = create_mock_otel_span(trace_id=trace_id, span_id=1, parent_id=None, start_time=5_000_000)

    # Mock get_active_spans_table_name to return None
    with mock.patch(
        "mlflow.tracing.processor.uc_table.get_active_spans_table_name", return_value=None
    ):
        processor = DatabricksUCTableSpanProcessor(span_exporter=mock.MagicMock())
        with pytest.raises(MlflowException, match="Unity Catalog spans table name is not set"):
            processor.on_start(span)

    # Check that trace was still created in trace manager
    trace_manager = InMemoryTraceManager.get_instance()
    traces = trace_manager._traces
    assert len(traces) == 0


def test_constructor_disables_metrics_export():
    mock_exporter = mock.MagicMock()
    processor = DatabricksUCTableSpanProcessor(span_exporter=mock_exporter)

    # The export_metrics should be False
    assert not processor._export_metrics


def test_trace_id_generation_with_uc_schema():
    trace_id = 12345
    span = create_mock_otel_span(trace_id=trace_id, span_id=1, parent_id=None, start_time=5_000_000)

    with (
        mock.patch(
            "mlflow.tracing.processor.uc_table.get_active_spans_table_name",
            return_value="catalog1.schema1.spans_table",
        ),
        mock.patch(
            "mlflow.tracing.processor.uc_table.generate_trace_id_v4",
            return_value="trace:/catalog1.schema1/12345",
        ) as mock_generate_trace_id,
    ):
        processor = DatabricksUCTableSpanProcessor(span_exporter=mock.MagicMock())
        processor.on_start(span)

        # Verify generate_trace_id_v4 was called with correct arguments
        mock_generate_trace_id.assert_called_once_with(span, "catalog1.schema1")


def test_on_end():
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
    span.set_inputs({"input1": "test input"})
    span.set_outputs({"output": "test output"})

    mock_exporter = mock.MagicMock()
    processor = DatabricksUCTableSpanProcessor(span_exporter=mock_exporter)

    processor.on_end(otel_span)

    # Verify span was exported
    mock_exporter.export.assert_called_once_with((otel_span,))


def test_trace_metadata_and_tags():
    trace_id = 12345
    span = create_mock_otel_span(trace_id=trace_id, span_id=1, parent_id=None, start_time=5_000_000)

    with mock.patch(
        "mlflow.tracing.processor.uc_table.get_active_spans_table_name",
        return_value="catalog1.schema1.spans_table",
    ):
        processor = DatabricksUCTableSpanProcessor(span_exporter=mock.MagicMock())
        processor.on_start(span)

    # Get the created trace
    trace_manager = InMemoryTraceManager.get_instance()
    traces = trace_manager._traces
    created_trace = list(traces.values())[0]
    trace_info = created_trace.info

    # Check that metadata and tags are present
    assert trace_info.trace_metadata is not None
    assert trace_info.tags is not None

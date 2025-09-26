from unittest import mock

import pytest

from mlflow.entities.span import LiveSpan
from mlflow.tracing.export.uc_table import DatabricksUCTableSpanExporter
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import generate_trace_id_v4

from tests.tracing.helper import (
    create_mock_otel_span,
    create_test_trace_info_with_uc_table,
)


def test_collect_mlflow_spans_for_export_uc_table():
    trace_manager = InMemoryTraceManager.get_instance()
    exporter = DatabricksUCTableSpanExporter()

    # Create spans with UC table configuration
    otel_span1 = create_mock_otel_span(trace_id=12345, span_id=1)
    otel_span2 = create_mock_otel_span(trace_id=12345, span_id=2)

    trace_id = generate_trace_id_v4(otel_span1, "catalog.schema")
    span1 = LiveSpan(otel_span1, trace_id)
    span2 = LiveSpan(otel_span2, trace_id)

    # Create trace info with UC table name
    trace_info = create_test_trace_info_with_uc_table(trace_id, "catalog", "schema", "spans")
    trace_manager.register_trace(otel_span1.context.trace_id, trace_info)
    trace_manager.register_span(span1)
    trace_manager.register_span(span2)

    spans_by_uc_table = exporter._collect_mlflow_spans_for_export(
        [otel_span1, otel_span2], trace_manager
    )

    assert len(spans_by_uc_table) == 1
    assert "catalog.schema.spans" in spans_by_uc_table
    assert len(spans_by_uc_table["catalog.schema.spans"]) == 2


@pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
def test_export_spans_to_uc_table(is_async, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", str(is_async))
    trace_manager = InMemoryTraceManager.get_instance()

    mock_client = mock.MagicMock()
    exporter = DatabricksUCTableSpanExporter()
    exporter._client = mock_client

    otel_span = create_mock_otel_span(trace_id=12345, span_id=1)
    trace_id = generate_trace_id_v4(otel_span, "catalog.schema")
    span = LiveSpan(otel_span, trace_id)

    # Create trace info with UC table
    trace_info = create_test_trace_info_with_uc_table(trace_id, "catalog", "schema", "spans")
    trace_manager.register_trace(otel_span.context.trace_id, trace_info)
    trace_manager.register_span(span)

    # Export the span
    exporter.export([otel_span])

    if is_async:
        # For async tests, we need to flush the specific exporter's queue
        exporter._async_queue.flush(terminate=True)

    # Verify UC table logging was called
    mock_client._log_spans_to_uc_table.assert_called_once_with("catalog.schema.spans", [span])


def test_log_trace_no_upload_data_for_uc_schema():
    mock_client = mock.MagicMock()

    # Mock trace info with UC schema
    mock_trace_info = mock.MagicMock()
    mock_trace_info.trace_location.uc_schema = mock.MagicMock()
    mock_client.start_trace.return_value = mock_trace_info

    mock_trace = mock.MagicMock()
    mock_trace.info = mock.MagicMock()

    mock_prompts = []

    exporter = DatabricksUCTableSpanExporter()
    exporter._client = mock_client

    with mock.patch("mlflow.tracing.export.mlflow_v3.add_size_stats_to_trace_metadata"):
        exporter._log_trace(mock_trace, mock_prompts)

        # Verify start_trace was called but _upload_trace_data was not
        mock_client.start_trace.assert_called_once_with(mock_trace.info)
        mock_client._upload_trace_data.assert_not_called()


def test_log_trace_no_log_spans_to_uc_table_if_no_uc_schema():
    mock_client = mock.MagicMock()

    # Mock trace info without UC schema
    mock_trace_info = mock.MagicMock()
    mock_trace_info.trace_location.uc_schema = None
    mock_client.start_trace.return_value = mock_trace_info

    mock_trace = mock.MagicMock()
    mock_trace.info = mock.MagicMock()
    mock_trace.data = mock.MagicMock()

    mock_prompts = []

    exporter = DatabricksUCTableSpanExporter()
    exporter._client = mock_client

    with mock.patch("mlflow.tracing.export.uc_table.add_size_stats_to_trace_metadata"):
        exporter._log_trace(mock_trace, mock_prompts)

        # Verify both start_trace and _upload_trace_data were called
        mock_client.start_trace.assert_called_once_with(mock_trace.info)
        mock_client._log_spans_to_uc_table.assert_not_called()

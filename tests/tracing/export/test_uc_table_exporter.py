from unittest import mock

import pytest

from mlflow.entities.span import Span
from mlflow.tracing.export.uc_table import DatabricksUCTableSpanExporter
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import generate_trace_id_v4

from tests.tracing.helper import (
    create_mock_otel_span,
    create_test_trace_info_with_uc_table,
)


@pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
def test_export_spans_to_uc_table(is_async, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", str(is_async))
    trace_manager = InMemoryTraceManager.get_instance()

    mock_client = mock.MagicMock()
    exporter = DatabricksUCTableSpanExporter()
    exporter._client = mock_client

    otel_span = create_mock_otel_span(trace_id=12345, span_id=1)
    trace_id = generate_trace_id_v4(otel_span, "catalog.schema")
    span = Span(otel_span)

    # Create trace info with UC table
    trace_info = create_test_trace_info_with_uc_table(trace_id, "catalog", "schema")
    trace_manager.register_trace(otel_span.context.trace_id, trace_info)
    trace_manager.register_span(span)

    # Export the span
    with mock.patch(
        "mlflow.tracing.export.uc_table.get_active_spans_table_name",
        return_value="catalog.schema.spans",
    ):
        exporter.export([otel_span])

    if is_async:
        # For async tests, we need to flush the specific exporter's queue
        exporter._async_queue.flush(terminate=True)

    # Verify UC table logging was called
    mock_client.log_spans.assert_called_once()
    args = mock_client.log_spans.call_args
    assert args[0][0] == "catalog.schema.spans"
    assert len(args[0][1]) == 1
    assert isinstance(args[0][1][0], Span)
    assert args[0][1][0].to_dict() == span.to_dict()


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

    with mock.patch("mlflow.tracing.utils.add_size_stats_to_trace_metadata"):
        exporter._log_trace(mock_trace, mock_prompts)

        # Verify start_trace was called but _upload_trace_data was not
        mock_client.start_trace.assert_called_once_with(mock_trace.info)
        mock_client._upload_trace_data.assert_not_called()


def test_log_trace_no_log_spans_if_no_uc_schema():
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

    with mock.patch("mlflow.tracing.utils.add_size_stats_to_trace_metadata"):
        exporter._log_trace(mock_trace, mock_prompts)

        # Verify both start_trace and _upload_trace_data were called
        mock_client.start_trace.assert_called_once_with(mock_trace.info)
        mock_client.log_spans.assert_not_called()

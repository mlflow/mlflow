from unittest import mock

import pytest

from mlflow.entities.span import Span
from mlflow.tracing.export.uc_table_with_otel import DatabricksUCTableWithOtelSpanExporter
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import generate_trace_id_v4

from tests.tracing.helper import (
    create_mock_otel_span,
    create_test_trace_info_with_uc_table_prefix,
)


@pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
def test_export_spans_via_otlp(is_async, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", str(is_async))
    trace_manager = InMemoryTraceManager.get_instance()

    mock_otlp_exporter = mock.MagicMock()
    mock_client = mock.MagicMock()

    exporter = DatabricksUCTableWithOtelSpanExporter(otlp_exporter=mock_otlp_exporter)
    exporter._client = mock_client

    otel_span = create_mock_otel_span(trace_id=12345, span_id=1)
    trace_id = generate_trace_id_v4(otel_span, "catalog.schema")
    span = Span(otel_span)

    # Create trace info with UC table prefix
    trace_info = create_test_trace_info_with_uc_table_prefix(
        trace_id, "catalog", "schema", "prefix_"
    )
    trace_manager.register_trace(otel_span.context.trace_id, trace_info)
    trace_manager.register_span(span)

    # Export the span
    exporter.export([otel_span])

    # Verify OTLP exporter was called with spans
    mock_otlp_exporter.export.assert_called_once_with([otel_span])


def test_export_spans_incrementally_calls_otlp_exporter():
    mock_otlp_exporter = mock.MagicMock()
    exporter = DatabricksUCTableWithOtelSpanExporter(otlp_exporter=mock_otlp_exporter)

    otel_spans = [
        create_mock_otel_span(trace_id=12345, span_id=1),
        create_mock_otel_span(trace_id=12345, span_id=2),
    ]

    exporter._export_spans_incrementally(otel_spans)

    mock_otlp_exporter.export.assert_called_once_with(otel_spans)


def test_export_spans_incrementally_handles_otlp_error():
    mock_otlp_exporter = mock.MagicMock()
    mock_otlp_exporter.export.side_effect = Exception("OTLP export failed")

    exporter = DatabricksUCTableWithOtelSpanExporter(otlp_exporter=mock_otlp_exporter)

    otel_spans = [create_mock_otel_span(trace_id=12345, span_id=1)]

    # Should not raise, just log warning
    exporter._export_spans_incrementally(otel_spans)

    # Second call should only log debug (not warning again)
    exporter._export_spans_incrementally(otel_spans)
    assert exporter._has_raised_span_export_error is True


def test_should_log_spans_to_artifacts_returns_false():
    mock_otlp_exporter = mock.MagicMock()
    exporter = DatabricksUCTableWithOtelSpanExporter(otlp_exporter=mock_otlp_exporter)

    mock_trace_info = mock.MagicMock()
    result = exporter._should_log_spans_to_artifacts(mock_trace_info)

    assert result is False


def test_log_trace_calls_start_trace_but_not_upload():
    mock_otlp_exporter = mock.MagicMock()
    mock_client = mock.MagicMock()

    # Mock trace info returned from start_trace
    mock_returned_trace_info = mock.MagicMock()
    mock_returned_trace_info.trace_location.uc_table_prefix = mock.MagicMock()
    mock_client.start_trace.return_value = mock_returned_trace_info

    mock_trace = mock.MagicMock()
    mock_trace.info = mock.MagicMock()

    exporter = DatabricksUCTableWithOtelSpanExporter(otlp_exporter=mock_otlp_exporter)
    exporter._client = mock_client

    with mock.patch("mlflow.tracing.utils.add_size_stats_to_trace_metadata"):
        exporter._log_trace(mock_trace, prompts=[])

        # Verify start_trace was called
        mock_client.start_trace.assert_called_once_with(mock_trace.info)

        # Verify _upload_trace_data was NOT called (spans go via OTLP)
        mock_client._upload_trace_data.assert_not_called()


def test_shutdown_calls_otlp_exporter_shutdown():
    mock_otlp_exporter = mock.MagicMock()
    exporter = DatabricksUCTableWithOtelSpanExporter(otlp_exporter=mock_otlp_exporter)

    exporter.shutdown()

    mock_otlp_exporter.shutdown.assert_called_once()


def test_force_flush_calls_otlp_exporter_force_flush():
    mock_otlp_exporter = mock.MagicMock()
    mock_otlp_exporter.force_flush.return_value = True

    exporter = DatabricksUCTableWithOtelSpanExporter(otlp_exporter=mock_otlp_exporter)

    result = exporter.force_flush(timeout_millis=5000)

    mock_otlp_exporter.force_flush.assert_called_once_with(5000)
    assert result is True


def test_should_enable_async_logging_reads_env_var(monkeypatch):
    mock_otlp_exporter = mock.MagicMock()

    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "true")
    exporter = DatabricksUCTableWithOtelSpanExporter(otlp_exporter=mock_otlp_exporter)
    assert exporter._should_enable_async_logging() is True

    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
    exporter2 = DatabricksUCTableWithOtelSpanExporter(otlp_exporter=mock_otlp_exporter)
    assert exporter2._should_enable_async_logging() is False


def test_constructor_accepts_tracking_uri():
    mock_otlp_exporter = mock.MagicMock()

    with mock.patch("mlflow.tracing.export.mlflow_v3.TracingClient") as mock_client_class:
        DatabricksUCTableWithOtelSpanExporter(
            otlp_exporter=mock_otlp_exporter,
            tracking_uri="databricks://my-workspace",
        )

        mock_client_class.assert_called_once_with("databricks://my-workspace")

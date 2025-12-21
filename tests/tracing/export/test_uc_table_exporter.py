import time
from concurrent.futures import ThreadPoolExecutor
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
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_SPAN_BATCH_SIZE", "1")  # no batch
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


def test_export_spans_batch_max_size(monkeypatch):
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_SPAN_BATCH_SIZE", "5")
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_INTERVAL_MILLIS", "10000")

    exporter = DatabricksUCTableSpanExporter()
    exporter._client = mock.MagicMock()
    with mock.patch(
        "mlflow.tracing.export.uc_table.get_active_spans_table_name",
        return_value="catalog.schema.spans",
    ):
        exporter._export_spans_incrementally(
            [
                create_mock_otel_span(trace_id=12345, span_id=1),
                create_mock_otel_span(trace_id=12345, span_id=2),
                create_mock_otel_span(trace_id=12345, span_id=3),
                create_mock_otel_span(trace_id=12345, span_id=4),
            ]
        )
        exporter._client.log_spans.assert_not_called()

        exporter._export_spans_incrementally([create_mock_otel_span(trace_id=12345, span_id=5)])
        # NB: There can be a tiny delay once the batch becomes full and the worker thread
        # is interrupted by the threading event and activate the async queue. Flush has to
        # happen after the activation.
        time.sleep(1)
        exporter._async_queue.flush()
        exporter._client.log_spans.assert_called_once()
        location, spans = exporter._client.log_spans.call_args[0]
        assert location == "catalog.schema.spans"
        assert len(spans) == 5
        assert all(isinstance(span, Span) for span in spans)


def test_export_spans_batch_flush_on_interval(monkeypatch):
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_SPAN_BATCH_SIZE", "10")
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_INTERVAL_MILLIS", "1000")

    exporter = DatabricksUCTableSpanExporter()
    exporter._client = mock.MagicMock()

    otel_span = create_mock_otel_span(trace_id=12345, span_id=1)

    with mock.patch(
        "mlflow.tracing.export.uc_table.get_active_spans_table_name",
        return_value="catalog.schema.spans",
    ):
        exporter._export_spans_incrementally([otel_span])

    # Allow the batcher's interval timer to fire
    time.sleep(1.5)

    exporter._client.log_spans.assert_called_once()
    location, spans = exporter._client.log_spans.call_args[0]
    assert location == "catalog.schema.spans"
    assert len(spans) == 1


def test_export_spans_batch_shutdown(monkeypatch):
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_SPAN_BATCH_SIZE", "10")
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_INTERVAL_MILLIS", "1000")

    exporter = DatabricksUCTableSpanExporter()
    exporter._client = mock.MagicMock()

    with mock.patch(
        "mlflow.tracing.export.uc_table.get_active_spans_table_name",
        return_value="catalog.schema.spans",
    ):
        exporter._export_spans_incrementally(
            [
                create_mock_otel_span(trace_id=12345, span_id=1),
                create_mock_otel_span(trace_id=12345, span_id=2),
                create_mock_otel_span(trace_id=12345, span_id=3),
            ]
        )

    exporter.flush()
    exporter._client.log_spans.assert_called_once()
    location, spans = exporter._client.log_spans.call_args[0]
    assert location == "catalog.schema.spans"
    assert len(spans) == 3


def test_export_spans_batch_thread_safety(monkeypatch):
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_SPAN_BATCH_SIZE", "10")
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_INTERVAL_MILLIS", "1000")

    exporter = DatabricksUCTableSpanExporter()
    exporter._client = mock.MagicMock()

    def _generate_spans():
        exporter._export_spans_incrementally(
            [create_mock_otel_span(trace_id=12345, span_id=i) for i in range(5)]
        )

    with mock.patch(
        "mlflow.tracing.export.uc_table.get_active_spans_table_name",
        return_value="catalog.schema.spans",
    ):
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_generate_spans) for _ in range(5)]
            for future in futures:
                future.result()

        exporter.flush()

        assert exporter._client.log_spans.call_count == 3
        for i in range(3):
            location, spans = exporter._client.log_spans.call_args_list[i][0]
            assert location == "catalog.schema.spans"
            assert len(spans) == 10 if i < 2 else 5, f"Batch {i} had {len(spans)} spans"


def test_export_spans_batch_split_spans_by_location(monkeypatch):
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_SPAN_BATCH_SIZE", "10")
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_INTERVAL_MILLIS", "1000")

    exporter = DatabricksUCTableSpanExporter()
    exporter._client = mock.MagicMock()

    with mock.patch(
        "mlflow.tracing.export.uc_table.get_active_spans_table_name",
        return_value="catalog.schema.table_1",
    ):
        exporter._export_spans_incrementally(
            [
                create_mock_otel_span(trace_id=12345, span_id=1),
                create_mock_otel_span(trace_id=12345, span_id=2),
            ]
        )

    with mock.patch(
        "mlflow.tracing.export.uc_table.get_active_spans_table_name",
        return_value="catalog.schema.table_2",
    ):
        exporter._export_spans_incrementally(
            [
                create_mock_otel_span(trace_id=12345, span_id=3),
                create_mock_otel_span(trace_id=12345, span_id=4),
                create_mock_otel_span(trace_id=12345, span_id=5),
            ]
        )

    exporter.flush()

    assert exporter._client.log_spans.call_count == 2
    location, spans = exporter._client.log_spans.call_args_list[0][0]
    assert location == "catalog.schema.table_1"
    assert len(spans) == 2
    location, spans = exporter._client.log_spans.call_args_list[1][0]
    assert location == "catalog.schema.table_2"
    assert len(spans) == 3


def test_at_exit_callback_registered_in_correct_order(monkeypatch):
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_SPAN_BATCH_SIZE", "10")
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_INTERVAL_MILLIS", "1000")

    # This test validates that the two atexit callbacks are registered in the correct order.
    # AsyncTraceExportQueue must be shut down AFTER SpanBatcher. Since atexit executes callbacks in
    # last-in-first-out order, we must register the callback for AsyncTraceExportQueue first.
    # https://docs.python.org/3/library/atexit.html#atexit.register
    with mock.patch("atexit.register") as mock_atexit:
        DatabricksUCTableSpanExporter()

    assert mock_atexit.call_count == 2
    handlers = [call[0][0] for call in mock_atexit.call_args_list]
    assert len(handlers) == 2
    assert handlers[0].__self__.__class__.__name__ == "AsyncTraceExportQueue"
    assert handlers[1].__self__.__class__.__name__ == "SpanBatcher"

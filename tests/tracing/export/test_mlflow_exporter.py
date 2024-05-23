from unittest.mock import MagicMock

from mlflow.entities import LiveSpan
from mlflow.tracing.export.mlflow import MlflowSpanExporter
from mlflow.tracing.fluent import TRACE_BUFFER
from mlflow.tracing.trace_manager import InMemoryTraceManager

from tests.tracing.helper import create_mock_otel_span, create_test_trace_info


def test_export():
    trace_id = 12345
    request_id = f"tr-{trace_id}"
    otel_span = create_mock_otel_span(
        trace_id=trace_id,
        span_id=1,
        parent_id=None,
        start_time=0,
        end_time=1_000_000,  # nano seconds
    )
    span = LiveSpan(otel_span, request_id=request_id)
    span.set_inputs({"input1": "very long input" * 100})
    span.set_outputs({"output": "very long output" * 100})

    trace_info = create_test_trace_info(request_id, 0)
    trace_manager = InMemoryTraceManager.get_instance()
    trace_manager.register_trace(trace_id, trace_info)
    trace_manager.register_span(span)

    # Non-root span should be ignored
    non_root_otel_span = create_mock_otel_span(trace_id=trace_id, span_id=2, parent_id=1)
    child_span = LiveSpan(non_root_otel_span, request_id=request_id)
    trace_manager.register_span(child_span)

    # Invalid span should be also ignored
    invalid_otel_span = create_mock_otel_span(trace_id=23456, span_id=1)

    mock_client = MagicMock()
    mock_display = MagicMock()
    exporter = MlflowSpanExporter(mock_client, mock_display)

    exporter.export([otel_span, non_root_otel_span, invalid_otel_span])

    # Spans should be cleared from the trace manager
    assert len(exporter._trace_manager._traces) == 0

    # Trace should be added to the in-memory buffer and displayed
    assert len(TRACE_BUFFER) == 1
    mock_display.display_traces.assert_called_once()

    # Trace should be logged
    assert mock_client._upload_trace_data.call_count == 1
    logged_trace_info, logged_trace_data = mock_client._upload_trace_data.call_args[0]
    assert trace_info == logged_trace_info
    assert len(logged_trace_data.spans) == 2
    mock_client._upload_ended_trace_info.assert_called_once_with(trace_info)

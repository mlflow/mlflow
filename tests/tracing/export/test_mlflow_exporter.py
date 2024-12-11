import threading
from concurrent.futures import ThreadPoolExecutor
from unittest import mock
from unittest.mock import MagicMock

from mlflow.entities import LiveSpan
from mlflow.tracing.export.mlflow import AsyncTraceExportQueue, MlflowSpanExporter, Task
from mlflow.tracing.fluent import TRACE_BUFFER
from mlflow.tracing.trace_manager import InMemoryTraceManager

from tests.tracing.helper import create_mock_otel_span, create_test_trace_info


def test_export(async_logging_enabled):
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

    if async_logging_enabled:
        exporter._async_queue.flush(terminate=True)

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


def test_async_queue_handle_tasks():
    mock_client = MagicMock()
    queue = AsyncTraceExportQueue(mock_client)

    counter = 0

    def _increment(delta):
        nonlocal counter
        counter += delta

    for _ in range(10):
        task = Task(handler=_increment, args=(1,))
        queue.put(task)

    queue.flush(terminate=True)
    assert counter == 10
    assert len(queue._unprocessed_tasks) == 0


@mock.patch("atexit.register")
def test_async_queue_activate_thread_safe(mock_atexit):
    mock_client = MagicMock()
    queue = AsyncTraceExportQueue(mock_client)

    def _count_threads():
        main_thread = threading.main_thread()
        return sum(
            t.is_alive()
            for t in threading.enumerate()
            if t is not main_thread and t.getName().startswith("MLflowTraceLogging")
        )

    # 1. Validate activation
    with ThreadPoolExecutor(max_workers=10) as executor:
        for _ in range(10):
            executor.submit(queue.activate)

    assert queue.is_active()
    assert _count_threads() > 0  # Logging thread + max 5 worker threads
    mock_atexit.assert_called_once()
    mock_atexit.reset_mock()

    # 2. Validate flush (continue)
    queue.flush(terminate=False)
    assert queue.is_active()
    assert _count_threads() > 0  # New threads should be created
    mock_atexit.assert_not_called()  # Exit callback should not be registered again

    # 3. Validate flush with termination
    with ThreadPoolExecutor(max_workers=10) as executor:
        for _ in range(10):
            executor.submit(queue.flush(terminate=True))

    assert not queue.is_active()
    assert _count_threads() == 0

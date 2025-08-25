import time
from threading import Thread

from mlflow.entities import LiveSpan, Span
from mlflow.entities.span_status import SpanStatusCode
from mlflow.tracing.trace_manager import InMemoryTraceManager, ManagerTrace

from tests.tracing.helper import create_mock_otel_span, create_test_trace_info


def test_aggregator_singleton():
    obj1 = InMemoryTraceManager.get_instance()
    obj2 = InMemoryTraceManager.get_instance()
    assert obj1 is obj2


def test_add_spans():
    trace_manager = InMemoryTraceManager.get_instance()

    # Add a new trace info
    request_id_1 = "tr-1"
    trace_id_1 = 12345
    trace_manager.register_trace(trace_id_1, create_test_trace_info(request_id_1, "test_1"))

    # Add a span for a new trace
    span_1_1 = _create_test_span(request_id_1, trace_id_1, span_id=1)
    trace_manager.register_span(span_1_1)

    assert request_id_1 in trace_manager._traces
    assert len(trace_manager._traces[request_id_1].span_dict) == 1

    # Add more spans to the same trace
    span_1_1_1 = _create_test_span(request_id_1, trace_id_1, span_id=2, parent_id=1)
    span_1_1_2 = _create_test_span(request_id_1, trace_id_1, span_id=3, parent_id=1)
    trace_manager.register_span(span_1_1_1)
    trace_manager.register_span(span_1_1_2)

    assert len(trace_manager._traces[request_id_1].span_dict) == 3

    # Add a span for another trace under the different experiment
    request_id_2 = "tr-2"
    trace_id_2 = 67890
    trace_manager.register_trace(trace_id_2, create_test_trace_info(request_id_2, "test_2"))

    span_2_1 = _create_test_span(request_id_2, trace_id_2, span_id=1)
    span_2_1_1 = _create_test_span(request_id_2, trace_id_2, span_id=2, parent_id=1)
    trace_manager.register_span(span_2_1)
    trace_manager.register_span(span_2_1_1)

    assert request_id_2 in trace_manager._traces
    assert len(trace_manager._traces[request_id_2].span_dict) == 2

    # Pop the trace data
    manager_trace = trace_manager.pop_trace(trace_id_1)
    assert isinstance(manager_trace, ManagerTrace)
    assert manager_trace.trace.info.request_id == request_id_1
    assert len(manager_trace.trace.data.spans) == 3
    assert request_id_1 not in trace_manager._traces

    # Convert to MLflow trace to check immutable spans
    trace = manager_trace.trace
    assert isinstance(trace.data.spans[0], Span)
    assert not isinstance(trace.data.spans[0], LiveSpan)

    manager_trace = trace_manager.pop_trace(trace_id_2)
    assert isinstance(manager_trace, ManagerTrace)
    assert manager_trace.trace.info.request_id == request_id_2
    assert len(manager_trace.trace.data.spans) == 2
    assert request_id_2 not in trace_manager._traces

    # Pop a trace that does not exist
    assert trace_manager.pop_trace(90123) is None


def test_add_and_pop_span_thread_safety():
    trace_manager = InMemoryTraceManager.get_instance()

    # Add spans from 10 different threads to 5 different traces
    trace_ids = list(range(5))
    num_threads = 10

    for trace_id in trace_ids:
        trace_manager.register_trace(trace_id, create_test_trace_info(f"tr-{trace_id}", "test"))

    def add_spans(thread_id):
        for trace_id in trace_ids:
            trace_manager.register_span(
                _create_test_span(f"tr-{trace_id}", trace_id=trace_id, span_id=thread_id)
            )

    threads = [Thread(target=add_spans, args=[i]) for i in range(num_threads)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for trace_id in trace_ids:
        manager_trace = trace_manager.pop_trace(trace_id)
        assert manager_trace is not None
        assert manager_trace.trace.info.request_id == f"tr-{trace_id}"
        assert len(manager_trace.trace.data.spans) == num_threads


def test_traces_buffer_expires_after_ttl(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACE_BUFFER_TTL_SECONDS", "1")

    trace_manager = InMemoryTraceManager.get_instance()
    request_id_1 = "tr-1"
    trace_manager.register_trace(12345, create_test_trace_info(request_id_1, "test"))

    span_1_1 = _create_test_span(request_id_1)
    trace_manager.register_span(span_1_1)

    assert request_id_1 in trace_manager._traces
    assert len(trace_manager._traces[request_id_1].span_dict) == 1

    time.sleep(1)

    assert request_id_1 not in trace_manager._traces

    # Clear singleton instance again to avoid side effects to other tests
    InMemoryTraceManager._instance = None


def test_traces_buffer_expires_and_log_when_timeout_is_set(monkeypatch):
    # Setting MLFLOW_TRACE_TIMEOUT_SECONDS let MLflow to periodically check the
    # expired traces and log expired ones to the backend.
    monkeypatch.setenv("MLFLOW_TRACE_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("MLFLOW_TRACE_TTL_CHECK_INTERVAL_SECONDS", "1")

    trace_manager = InMemoryTraceManager.get_instance()
    request_id_1 = "tr-1"
    trace_info = create_test_trace_info(request_id_1, "test")
    trace_manager.register_trace(12345, trace_info)

    span_1_1 = _create_test_span(request_id_1)
    trace_manager.register_span(span_1_1)

    assert trace_manager._traces.get(request_id_1) is not None
    assert len(trace_manager._traces[request_id_1].span_dict) == 1

    assert request_id_1 in trace_manager._traces

    time.sleep(3)

    assert request_id_1 not in trace_manager._traces
    assert span_1_1.status.status_code == SpanStatusCode.ERROR


def test_traces_buffer_max_size_limit(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACE_BUFFER_MAX_SIZE", "1")

    trace_manager = InMemoryTraceManager.get_instance()
    request_id_1 = "tr-1"
    trace_manager.register_trace(12345, create_test_trace_info(request_id_1, "experiment"))

    span_1_1 = _create_test_span(request_id_1)
    trace_manager.register_span(span_1_1)

    assert request_id_1 in trace_manager._traces
    assert len(trace_manager._traces) == 1

    request_id_2 = "tr-2"
    trace_manager.register_trace(67890, create_test_trace_info(request_id_2, "experiment"))
    span_2_1 = _create_test_span(request_id_2)
    trace_manager.register_span(span_2_1)

    assert request_id_1 not in trace_manager._traces
    assert request_id_2 in trace_manager._traces
    assert len(trace_manager._traces) == 1

    # Clear singleton instance again to avoid side effects to other tests
    InMemoryTraceManager._instance = None


def test_get_span_from_id():
    trace_manager = InMemoryTraceManager.get_instance()
    request_id_1 = "tr-1"
    trace_manager.register_trace(12345, create_test_trace_info(request_id_1, "test"))
    span_1_1 = _create_test_span(request_id_1, trace_id=111, span_id=1)
    span_1_2 = _create_test_span(request_id_1, trace_id=111, span_id=2, parent_id=1)

    request_id_2 = "tr-2"
    trace_manager.register_trace(67890, create_test_trace_info(request_id_2, "test"))
    span_2_1 = _create_test_span(request_id_2, trace_id=222, span_id=1)
    span_2_2 = _create_test_span(request_id_2, trace_id=222, span_id=2, parent_id=1)

    # Add a span for a new trace
    trace_manager.register_span(span_1_1)
    trace_manager.register_span(span_1_2)
    trace_manager.register_span(span_2_1)
    trace_manager.register_span(span_2_2)

    assert trace_manager.get_span_from_id(request_id_1, span_1_1.span_id) == span_1_1
    assert trace_manager.get_span_from_id(request_id_2, span_2_2.span_id) == span_2_2


def test_get_root_span_id():
    trace_manager = InMemoryTraceManager.get_instance()

    request_id_1 = "tr-1"
    trace_manager.register_trace(12345, create_test_trace_info(request_id_1, "test"))
    span_1_1 = _create_test_span(request_id_1, span_id=1)
    span_1_2 = _create_test_span(request_id_1, span_id=2, parent_id=1)

    # Add a span for a new trace
    trace_manager.register_span(span_1_1)
    trace_manager.register_span(span_1_2)

    assert trace_manager.get_root_span_id(request_id_1) == span_1_1.span_id

    # Non-existing trace
    assert trace_manager.get_root_span_id("tr-2") is None


def _create_test_span(
    request_id="tr-12345",
    trace_id: int = 12345,
    span_id: int = 123,
    parent_id: int | None = None,
    start_time=None,
    end_time=None,
):
    mock_otel_span = create_mock_otel_span(
        trace_id=trace_id,
        span_id=span_id,
        parent_id=parent_id,
        start_time=start_time,
        end_time=end_time,
    )

    span = LiveSpan(mock_otel_span, request_id)
    span.set_status("OK")
    return span

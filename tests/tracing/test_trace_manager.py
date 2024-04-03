import time
from threading import Thread
from unittest import mock

import pytest
from opentelemetry import trace as trace_api

from mlflow.entities import Trace
from mlflow.exceptions import MlflowException
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.types.wrapper import MlflowSpanWrapper


def test_aggregator_singleton():
    obj1 = InMemoryTraceManager.get_instance()
    obj2 = InMemoryTraceManager.get_instance()
    assert obj1 is obj2


def test_add_spans():
    trace_manager = InMemoryTraceManager.get_instance()

    request_id_1 = "trace_1"
    span_1_1 = _create_test_span(request_id_1, "span_1_1")
    span_1_1_1 = _create_test_span(request_id_1, "span_1_1_1", parent_span_id="span_1_1")
    span_1_1_2 = _create_test_span(request_id_1, "span_1_1_2", parent_span_id="span_1_1")

    # Add a span for a new trace
    trace_manager.add_or_update_span(span_1_1)

    assert request_id_1 in trace_manager._traces
    assert len(trace_manager._traces[request_id_1].span_dict) == 1

    # Add more spans to the same trace
    trace_manager.add_or_update_span(span_1_1_1)
    trace_manager.add_or_update_span(span_1_1_2)

    assert len(trace_manager._traces[request_id_1].span_dict) == 3

    # Add a span for another trace
    request_id_2 = "trace_2"
    span_2_1 = _create_test_span(request_id_2, "span_2_1")
    span_2_1_1 = _create_test_span(request_id_2, "span_2_1_1", parent_span_id="span_2_1")

    trace_manager.add_or_update_span(span_2_1)
    trace_manager.add_or_update_span(span_2_1_1)

    assert request_id_2 in trace_manager._traces
    assert len(trace_manager._traces[request_id_2].span_dict) == 2

    # Pop the trace data
    trace = trace_manager.pop_trace(request_id_1)
    assert isinstance(trace, Trace)
    assert len(trace.trace_data.spans) == 3
    assert request_id_1 not in trace_manager._traces

    trace = trace_manager.pop_trace(request_id_2)
    assert isinstance(trace, Trace)
    assert len(trace.trace_data.spans) == 2
    assert request_id_2 not in trace_manager._traces

    # Pop a trace that does not exist
    assert trace_manager.pop_trace("trace_3") is None


def test_start_detached_span():
    trace_manager = InMemoryTraceManager.get_instance()

    # Root span will create a new trace
    root_span = trace_manager.start_detached_span(name="root_span")
    request_id = root_span.request_id
    assert len(trace_manager._traces) == 1
    assert trace_manager.get_root_span_id(request_id) == root_span.span_id

    # Child span will be added to the existing trace
    child_span = trace_manager.start_detached_span(
        name="child_span", request_id=request_id, parent_span_id=root_span.span_id
    )

    assert len(trace_manager._traces) == 1
    assert trace_manager.get_span_from_id(request_id, span_id=child_span.span_id) == child_span


def test_add_and_pop_span_thread_safety():
    trace_manager = InMemoryTraceManager.get_instance()

    # Add spans from 10 different threads to 5 different traces
    request_ids = [f"trace_{i}" for i in range(5)]
    num_threads = 10

    def add_spans(thread_id):
        for request_id in request_ids:
            trace_manager.add_or_update_span(_create_test_span(request_id, f"span_{thread_id}"))

    threads = [Thread(target=add_spans, args=[i]) for i in range(num_threads)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for request_id in request_ids:
        trace = trace_manager.pop_trace(request_id)
        assert trace is not None
        assert trace.trace_info.request_id == request_id
        assert len(trace.trace_data.spans) == num_threads


def test_traces_buffer_expires_after_ttl(monkeypatch):
    # Clear singleton instance to patch TTL
    InMemoryTraceManager._instance = None
    monkeypatch.setenv("MLFLOW_TRACE_BUFFER_TTL_SECONDS", "1")

    trace_manager = InMemoryTraceManager.get_instance()

    trace_id_1 = "trace_1"
    span_1_1 = _create_test_span(trace_id_1, "span")
    trace_manager.add_or_update_span(span_1_1)

    assert trace_id_1 in trace_manager._traces
    assert len(trace_manager._traces[trace_id_1].span_dict) == 1

    time.sleep(1)

    assert trace_id_1 not in trace_manager._traces

    # Clear singleton instance again to avoid side effects to other tests
    InMemoryTraceManager._instance = None


def test_traces_buffer_max_size_limit(monkeypatch):
    # Clear singleton instance to patch buffer size
    InMemoryTraceManager._instance = None
    monkeypatch.setenv("MLFLOW_TRACE_BUFFER_MAX_SIZE", "1")

    trace_manager = InMemoryTraceManager.get_instance()

    trace_id_1 = "trace_1"
    span_1_1 = _create_test_span(trace_id_1, "span")
    trace_manager.add_or_update_span(span_1_1)

    assert trace_id_1 in trace_manager._traces
    assert len(trace_manager._traces) == 1

    trace_id_2 = "trace_2"
    span_2_1 = _create_test_span(trace_id_2, "span")
    trace_manager.add_or_update_span(span_2_1)

    assert trace_id_1 not in trace_manager._traces
    assert trace_id_2 in trace_manager._traces
    assert len(trace_manager._traces) == 1

    # Clear singleton instance again to avoid side effects to other tests
    InMemoryTraceManager._instance = None


def test_get_span_from_id():
    trace_manager = InMemoryTraceManager.get_instance()

    request_id_1 = "trace_1"
    span_1_1 = _create_test_span(request_id_1, "span")
    span_1_2 = _create_test_span(request_id_1, "child_span", parent_span_id=span_1_1.span_id)

    request_id_2 = "trace_2"
    span_2_1 = _create_test_span(request_id_2, "span")
    span_2_2 = _create_test_span(request_id_2, "child_span", parent_span_id=span_2_1.span_id)

    # Add a span for a new trace
    trace_manager.add_or_update_span(span_1_1)
    trace_manager.add_or_update_span(span_1_2)
    trace_manager.add_or_update_span(span_2_1)
    trace_manager.add_or_update_span(span_2_2)

    assert trace_manager.get_span_from_id(request_id_1, "span") == span_1_1
    assert trace_manager.get_span_from_id(request_id_2, "child_span") == span_2_2


def test_ger_root_span_id():
    trace_manager = InMemoryTraceManager.get_instance()

    request_id_1 = "trace_1"
    span_1_1 = _create_test_span(request_id_1, "span")
    span_1_2 = _create_test_span(request_id_1, "child_span", parent_span_id=span_1_1.span_id)

    # Add a span for a new trace
    trace_manager.add_or_update_span(span_1_1)
    trace_manager.add_or_update_span(span_1_2)

    assert trace_manager.get_root_span_id(request_id_1) == "span"

    # Non-existing trace
    assert trace_manager.get_root_span_id("trace_2") is None


def test_set_trace_tag():
    trace_manager = InMemoryTraceManager.get_instance()

    request_id = "trace_1"
    span = _create_test_span(request_id, "span")
    trace_manager.add_or_update_span(span)

    trace_manager.set_trace_tag(request_id, "foo", "bar")
    assert trace_manager.get_trace_info(request_id).tags == {"foo": "bar"}


def test_set_trace_tag_raises_when_trace_not_found():
    trace_manager = InMemoryTraceManager.get_instance()

    with pytest.raises(MlflowException, match="Trace with ID test not found."):
        trace_manager.set_trace_tag("test", "foo", "bar")


def _create_test_span(request_id, span_id, parent_span_id=None, start_time=None, end_time=None):
    if start_time is None:
        start_time = time.time_ns()
    if end_time is None:
        end_time = time.time_ns()

    mock_span = mock.MagicMock()
    mock_span.get_span_context().trace_id = request_id
    mock_span.get_span_context().span_id = span_id
    mock_span.parent.span_id = parent_span_id
    mock_span.start_time = start_time
    mock_span.end_time = end_time
    mock_span.name = "test_span"
    mock_span.status.status_code = trace_api.StatusCode.OK
    mock_span.status.description = ""

    return MlflowSpanWrapper(mock_span)

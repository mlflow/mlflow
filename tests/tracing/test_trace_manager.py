import time
from threading import Thread
from unittest import mock

import pytest

import mlflow
from mlflow.entities import Trace
from mlflow.exceptions import MlflowException
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.types.wrapper import MlflowSpanWrapper, NoOpMlflowSpanWrapper

from tests.tracing.helper import create_mock_otel_span


def test_aggregator_singleton():
    obj1 = InMemoryTraceManager.get_instance()
    obj2 = InMemoryTraceManager.get_instance()
    assert obj1 is obj2


def test_add_spans():
    trace_manager = InMemoryTraceManager.get_instance()

    exp_id_1 = mlflow.set_experiment("test_experiment_1").experiment_id
    request_id_1 = "tr-1"
    trace_id_1 = 12345
    span_1_1 = _create_test_span(request_id_1, trace_id_1, span_id=1)
    span_1_1_1 = _create_test_span(request_id_1, trace_id_1, span_id=2, parent_id=1)
    span_1_1_2 = _create_test_span(request_id_1, trace_id_1, span_id=3, parent_id=1)

    # Add a span for a new trace
    trace_manager.add_or_update_span(span_1_1)

    assert request_id_1 in trace_manager._traces
    assert len(trace_manager._traces[request_id_1].span_dict) == 1

    # Add more spans to the same trace
    trace_manager.add_or_update_span(span_1_1_1)
    trace_manager.add_or_update_span(span_1_1_2)

    assert len(trace_manager._traces[request_id_1].span_dict) == 3

    # Add a span for another trace under the different experiment
    exp_id_2 = mlflow.set_experiment("test_experiment_2").experiment_id
    request_id_2 = "tr-2"
    trace_id_2 = 67890
    span_2_1 = _create_test_span(request_id_2, trace_id_2, span_id=1)
    span_2_1_1 = _create_test_span(request_id_2, trace_id_2, span_id=2, parent_id=1)

    trace_manager.add_or_update_span(span_2_1)
    trace_manager.add_or_update_span(span_2_1_1)

    assert request_id_2 in trace_manager._traces
    assert len(trace_manager._traces[request_id_2].span_dict) == 2

    # Pop the trace data
    trace = trace_manager.pop_trace(request_id_1)
    assert isinstance(trace, Trace)
    assert trace.info.request_id == request_id_1
    assert trace.info.experiment_id == exp_id_1
    assert len(trace.data.spans) == 3
    assert request_id_1 not in trace_manager._traces

    trace = trace_manager.pop_trace(request_id_2)
    assert isinstance(trace, Trace)
    assert trace.info.request_id == request_id_2
    assert trace.info.experiment_id == exp_id_2
    assert len(trace.data.spans) == 2
    assert request_id_2 not in trace_manager._traces

    # Pop a trace that does not exist
    assert trace_manager.pop_trace("tr-3") is None


def test_start_detached_span():
    trace_manager = InMemoryTraceManager.get_instance()

    # Root span will create a new trace
    root_span = trace_manager.start_detached_span(name="root_span")
    request_id = root_span.request_id
    assert len(trace_manager._traces) == 1
    assert trace_manager.get_root_span_id(request_id) == root_span.span_id

    # Child span will be added to the existing trace
    child_span = trace_manager.start_detached_span(
        name="child_span", request_id=request_id, parent_id=root_span.span_id
    )

    assert len(trace_manager._traces) == 1
    assert trace_manager.get_span_from_id(request_id, span_id=child_span.span_id) == child_span


def test_start_detached_span_show_warning_when_parent_not_found(caplog):
    trace_manager = InMemoryTraceManager.get_instance()

    with mock.patch("mlflow.tracing.trace_manager._logger") as mock_logger:
        span = trace_manager.start_detached_span(name="root_span", parent_id="not_found")

    assert isinstance(span, NoOpMlflowSpanWrapper)
    warning_message = mock_logger.warning.call_args[0][0]
    assert "Parent span with ID 'not_found' not found." in warning_message


def test_add_and_pop_span_thread_safety():
    trace_manager = InMemoryTraceManager.get_instance()

    # Add spans from 10 different threads to 5 different traces
    trace_ids = [i for i in range(5)]
    num_threads = 10

    def add_spans(thread_id):
        for trace_id in trace_ids:
            trace_manager.add_or_update_span(_create_test_span(f"tr-{trace_id}", trace_id=trace_id, span_id=thread_id))

    threads = [Thread(target=add_spans, args=[i]) for i in range(num_threads)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for trace_id in trace_ids:
        trace = trace_manager.pop_trace(f"tr-{trace_id}")
        assert trace is not None
        assert trace.info.request_id == f"tr-{trace_id}"
        assert len(trace.data.spans) == num_threads


def test_traces_buffer_expires_after_ttl(monkeypatch):
    # Clear singleton instance to patch TTL
    InMemoryTraceManager._instance = None
    monkeypatch.setenv("MLFLOW_TRACE_BUFFER_TTL_SECONDS", "1")

    trace_manager = InMemoryTraceManager.get_instance()

    request_id_1 = "tr-1"
    span_1_1 = _create_test_span(request_id_1)
    trace_manager.add_or_update_span(span_1_1)

    assert request_id_1 in trace_manager._traces
    assert len(trace_manager._traces[request_id_1].span_dict) == 1

    time.sleep(1)

    assert request_id_1 not in trace_manager._traces

    # Clear singleton instance again to avoid side effects to other tests
    InMemoryTraceManager._instance = None


def test_traces_buffer_max_size_limit(monkeypatch):
    # Clear singleton instance to patch buffer size
    InMemoryTraceManager._instance = None
    monkeypatch.setenv("MLFLOW_TRACE_BUFFER_MAX_SIZE", "1")

    trace_manager = InMemoryTraceManager.get_instance()

    request_id_1 = "tr-1"
    span_1_1 = _create_test_span(request_id_1)
    trace_manager.add_or_update_span(span_1_1)

    assert request_id_1 in trace_manager._traces
    assert len(trace_manager._traces) == 1

    request_id_2 = "tr-2"
    span_2_1 = _create_test_span(request_id_2)
    trace_manager.add_or_update_span(span_2_1)

    assert request_id_1 not in trace_manager._traces
    assert request_id_2 in trace_manager._traces
    assert len(trace_manager._traces) == 1

    # Clear singleton instance again to avoid side effects to other tests
    InMemoryTraceManager._instance = None


def test_get_span_from_id():
    trace_manager = InMemoryTraceManager.get_instance()

    request_id_1 = "tr-1"
    span_1_1 = _create_test_span(request_id_1, trace_id=111, span_id=1)
    span_1_2 = _create_test_span(request_id_1, trace_id=111, span_id=2, parent_id=1)

    request_id_2 = "tr-2"
    span_2_1 = _create_test_span(request_id_2, trace_id=222, span_id=1)
    span_2_2 = _create_test_span(request_id_2, trace_id=222, span_id=2, parent_id=1)

    # Add a span for a new trace
    trace_manager.add_or_update_span(span_1_1)
    trace_manager.add_or_update_span(span_1_2)
    trace_manager.add_or_update_span(span_2_1)
    trace_manager.add_or_update_span(span_2_2)

    assert trace_manager.get_span_from_id(request_id_1, span_1_1.span_id) == span_1_1
    assert trace_manager.get_span_from_id(request_id_2, span_2_2.span_id) == span_2_2


def test_get_root_span_id():
    trace_manager = InMemoryTraceManager.get_instance()

    request_id_1 = "tr-1"
    span_1_1 = _create_test_span(request_id_1, span_id=1)
    span_1_2 = _create_test_span(request_id_1, span_id=2, parent_id=1)

    # Add a span for a new trace
    trace_manager.add_or_update_span(span_1_1)
    trace_manager.add_or_update_span(span_1_2)

    assert trace_manager.get_root_span_id(request_id_1) == span_1_1.span_id

    # Non-existing trace
    assert trace_manager.get_root_span_id("tr-2") is None


def test_set_trace_tag():
    trace_manager = InMemoryTraceManager.get_instance()

    request_id = "tr-1"
    span = _create_test_span(request_id)
    trace_manager.add_or_update_span(span)

    trace_manager.set_trace_tag(request_id, "foo", "bar")
    assert trace_manager.get_trace_info(request_id).tags == {"foo": "bar"}


def test_set_trace_tag_raises_when_trace_not_found():
    with pytest.raises(MlflowException, match="Trace with ID test not found."):
        InMemoryTraceManager.get_instance().set_trace_tag("test", "foo", "bar")


def test_delete_trace_tag():
    trace_manager = InMemoryTraceManager.get_instance()

    request_id = "tr-1"
    span = _create_test_span(request_id)
    trace_manager.add_or_update_span(span)

    trace_manager.set_trace_tag(request_id, "foo", "bar")
    trace_manager.delete_trace_tag(request_id, "foo")
    assert trace_manager.get_trace_info(request_id).tags == {}

    # Raise when tag not found
    with pytest.raises(MlflowException, match="Tag with key baz not found in trace with ID tr-1."):
        trace_manager.delete_trace_tag(request_id, "baz")


def test_delete_tag_raises_when_trace_not_found():
    with pytest.raises(MlflowException, match="Trace with ID test not found."):
        InMemoryTraceManager.get_instance().delete_trace_tag("test", "foo")


def _create_test_span(request_id="tr-12345",
                      trace_id: int=12345,
                      span_id: int=123,
                      parent_id: int=None,
                      start_time=None,
                      end_time=None):
    mock_otel_span = create_mock_otel_span(trace_id=trace_id,
                                           span_id=span_id,
                                           parent_id=parent_id,
                                           start_time=start_time,
                                           end_time=end_time)

    span = MlflowSpanWrapper(mock_otel_span, request_id=request_id)
    span.set_status("OK")
    return span

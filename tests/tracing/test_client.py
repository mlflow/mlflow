import time

import pytest

import mlflow
from mlflow.exceptions import MlflowException, RestException
from mlflow.tracing.client import TracingClient, _retry_if_trace_is_pending_export
from mlflow.tracing.export.async_export_queue import AsyncTraceExportQueue

from tests.tracing.helper import get_traces


def test_set_and_delete_trace_tag_on_logged_trace():
    client = TracingClient()

    with mlflow.start_span() as span:
        pass

    client.set_trace_tag(span.trace_id, "foo", "bar")

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.tags["foo"] == "bar"

    client.delete_trace_tag(span.trace_id, "foo")

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert "foo" not in trace.info.tags


def test_set_and_delete_trace_tag_on_active_trace():
    client = TracingClient()

    with mlflow.start_span() as span:
        trace_id = span.trace_id
        client.set_trace_tag(trace_id, "foo", "bar")
        client.set_trace_tag(trace_id, "baz", "qux")
        client.delete_trace_tag(trace_id, "baz")

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.tags["foo"] == "bar"
    assert "baz" not in trace.info.tags


def test_set_trace_tag_on_pending_trace(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_LOGGING", "true")

    original = TracingClient._upload_ended_trace_info

    def _slow_upload_ended_trace_info(*args, **kwargs):
        time.sleep(5)
        original(*args, **kwargs)

    monkeypatch.setattr(TracingClient, "_upload_ended_trace_info", _slow_upload_ended_trace_info)

    with mlflow.start_span() as span:
        pass

    # Trace is still pending export, not uploaded to the backend yet
    assert get_traces() == []

    client = TracingClient()
    client.set_trace_tag(span.trace_id, "foo", "bar")

    # Force the trace to be uploaded to the backend
    mlflow.flush_trace_async_logging()
    assert len(get_traces()) == 1
    trace = get_traces()[0]
    assert trace.info.tags["foo"] == "bar"


def test_retry_decorator_should_retry_when_queue_is_not_empty(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_LOGGING", "true")
    monkeypatch.setattr(AsyncTraceExportQueue, "is_empty", lambda: False)
    monkeypatch.setattr(mlflow.tracing.client, "_PENDING_TRACE_MAX_RETRY_COUNT", 2)

    calls = []

    @_retry_if_trace_is_pending_export
    def dummy_method(self, trace_id, key, value):
        calls.append((trace_id, key, value))
        raise RestException({"error_code": "INVALID_PARAMETER_VALUE", "message": "Not found"})

    with pytest.raises(MlflowException, match=r"Failed to call dummy_method API."):
        dummy_method(None, "trace_id", "key", "value")

    assert len(calls) == 2
    assert all(call == ("trace_id", "key", "value") for call in calls)


def test_retry_decorator_should_not_retry_when_queue_is_empty(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_LOGGING", "false")
    monkeypatch.setattr(AsyncTraceExportQueue, "is_empty", lambda: True)

    calls = []

    @_retry_if_trace_is_pending_export
    def dummy_method(self, trace_id, key, value):
        calls.append((trace_id, key, value))
        raise RestException({"error_code": "INVALID_PARAMETER_VALUE", "message": "Not found"})

    with pytest.raises(RestException, match=r"INVALID_PARAMETER_VALUE"):
        dummy_method(None, "trace_id", "key", "value")

    assert len(calls) == 1
    assert calls[0] == ("trace_id", "key", "value")


def test_retry_decorator_should_not_retry_other_exceptions(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_LOGGING", "false")
    monkeypatch.setattr(AsyncTraceExportQueue, "is_empty", lambda: False)

    calls = []

    @_retry_if_trace_is_pending_export
    def dummy_method(self, trace_id, key, value):
        calls.append((trace_id, key, value))
        raise ValueError("test")

    with pytest.raises(ValueError, match="test"):
        dummy_method(None, "trace_id", "key", "value")

    assert len(calls) == 1
    assert calls[0] == ("trace_id", "key", "value")

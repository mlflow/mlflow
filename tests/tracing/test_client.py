import re
import time
from unittest import mock

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

    with mock.patch.object(
        TracingClient,
        "_upload_ended_trace_info",
        side_effect=_slow_upload_ended_trace_info,
    ):
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


def test_retry_decorator_should_when_async_logging_is_disabled(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_LOGGING", "false")
    calls = []

    @_retry_if_trace_is_pending_export
    def dummy_method(trace_id, key, value):
        calls.append((trace_id, key, value))
        raise RestException({"error_code": "INVALID_PARAMETER_VALUE", "message": "Not found"})

    with pytest.raises(MlflowException, match=r"INVALID_PARAMETER_VALUE: Not found"):
        dummy_method("trace_id", "key", "value")

    assert len(calls) == 1
    assert calls[0] == ("trace_id", "key", "value")


@pytest.mark.parametrize(
    ("error", "should_retry"),
    [
        (RestException({"error_code": "INVALID_PARAMETER_VALUE", "message": "Not found"}), False),
        (RestException({"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "Not found"}), True),
        (
            RestException(
                {
                    "error_code": "INVALID_PARAMETER_VALUE",
                    "message": "Traces with ids (trace_id) do not exist",
                }
            ),
            True,
        ),
    ],
)
def test_retry_decorator_should_only_catch_correct_exceptions(monkeypatch, error, should_retry):
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_LOGGING", "false")
    calls = []

    @_retry_if_trace_is_pending_export
    def dummy_method(trace_id, key, value):
        calls.append((trace_id, key, value))
        if len(calls) == 1:
            raise error
        return

    with pytest.raises(error.__class__, match=re.escape(error.message)):
        dummy_method("trace_id", "key", "value")

    assert len(calls) == 1 if should_retry else 2
    assert calls[0] == ("trace_id", "key", "value")


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

import json
from unittest import mock
from unittest.mock import Mock

import pytest

import mlflow
from mlflow.entities.span_status import SpanStatus
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_status import TraceStatus
from mlflow.tracing.clients import get_trace_client


class MockIPython:
    def __init__(self):
        self.execution_count = 0

    def mock_run_cell(self):
        self.execution_count += 1


@pytest.fixture(autouse=True)
def mock_tracking_serving_client():
    with mock.patch(
        "mlflow.tracking._tracking_service.client.TrackingServiceClient.create_trace_info",
        return_value=TraceInfo(
            request_id="tr-1234",
            experiment_id="0",
            timestamp_ms=0,
            execution_time_ms=0,
            status=SpanStatus(TraceStatus.OK),
            request_metadata={},
            tags={"mlflow.artifactLocation": "test"},
        ),
    ), mock.patch(
        "mlflow.tracking._tracking_service.client.TrackingServiceClient._upload_trace_data",
    ):
        yield


def test_display_is_not_called_without_ipython(monkeypatch, create_trace):
    # in an IPython environment, the interactive shell will
    # be returned. however, for test purposes, just mock that
    # the value is not None.
    mock_display = Mock()
    monkeypatch.setattr("IPython.display.display", mock_display)

    client = get_trace_client()
    client.log_trace(create_trace("a"))
    assert mock_display.call_count == 0

    monkeypatch.setattr("IPython.get_ipython", lambda: MockIPython())
    client.log_trace(create_trace("b"))
    assert mock_display.call_count == 1


def test_ipython_client_only_logs_once_per_execution(monkeypatch, create_trace):
    mock_ipython = MockIPython()
    monkeypatch.setattr("IPython.get_ipython", lambda: mock_ipython)
    client = get_trace_client()

    mock_display_handle = Mock()
    mock_display = Mock(return_value=mock_display_handle)
    monkeypatch.setattr("IPython.display.display", mock_display)
    client.log_trace(create_trace("a"))
    client.log_trace(create_trace("b"))
    client.log_trace(create_trace("c"))

    # there should be one display and two updates
    assert mock_display.call_count == 1
    assert mock_display_handle.update.call_count == 2

    # after incrementing the execution count,
    # the next log should call display again
    mock_ipython.mock_run_cell()
    client.log_trace(create_trace("a"))
    assert mock_display.call_count == 2


def test_display_is_called_in_correct_functions(monkeypatch, create_trace):
    mock_ipython = MockIPython()
    monkeypatch.setattr("IPython.get_ipython", lambda: mock_ipython)
    client = get_trace_client()

    mock_display_handle = Mock()
    mock_display = Mock(return_value=mock_display_handle)
    monkeypatch.setattr("IPython.display.display", mock_display)
    client.log_trace(create_trace("a"))
    assert mock_display.call_count == 1

    mock_ipython.mock_run_cell()
    mlflow.get_traces(n=10)
    assert mock_display.call_count == 2

    class MockMlflowClient:
        def search_traces(self, *args, **kwargs):
            return [create_trace("a"), create_trace("b"), create_trace("c")]

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)

    mock_ipython.mock_run_cell()
    mlflow.search_traces(["123"])
    assert mock_display.call_count == 3


def test_display_deduplicates_traces(monkeypatch, create_trace):
    mock_ipython = MockIPython()
    monkeypatch.setattr("IPython.get_ipython", lambda: mock_ipython)
    client = get_trace_client()

    mock_display_handle = Mock()
    mock_display = Mock(return_value=mock_display_handle)
    monkeypatch.setattr("IPython.display.display", mock_display)

    trace_a = create_trace("a")
    trace_b = create_trace("b")
    trace_c = create_trace("c")

    # 3 traces are created, and the same 3 traces
    # are returned by `get_traces()`. the display
    # client should dedupe these and only display
    # 3 traces (not 6).
    client.log_trace(trace_a)
    client.log_trace(trace_b)
    client.log_trace(trace_c)

    mlflow.get_traces(n=3)

    expected = [trace_a, trace_b, trace_c]

    assert mock_display.call_count == 1
    assert mock_display_handle.update.call_count == 3
    assert mock_display_handle.update.call_args[0][0] == {
        "application/databricks.mlflow.trace": json.dumps(
            [json.loads(t.to_json()) for t in expected]
        ),
        "text/plain": expected.__repr__(),
    }

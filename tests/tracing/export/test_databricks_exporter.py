import base64
import os
import time
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest

import mlflow
from mlflow.entities.span_event import SpanEvent
from mlflow.tracing.destination import Databricks
from mlflow.tracing.provider import _get_trace_exporter
from mlflow.tracking.fluent import _get_experiment_id

_EXPERIMENT_ID = "dummy-experiment-id"


@mlflow.trace
def _predict(x: str) -> str:
    with mlflow.start_span(name="child") as child_span:
        child_span.set_inputs("dummy")
        child_span.add_event(SpanEvent(name="child_event", attributes={"attr1": "val1"}))
    mlflow.update_current_trace(tags={"foo": "bar"})
    return x + "!"


def _flush_async_logging():
    exporter = _get_trace_exporter()
    assert hasattr(exporter, "_async_queue"), "Async queue is not initialized"
    exporter._async_queue.flush(terminate=True)


@pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
@pytest.mark.parametrize("experiment_id", [None, _EXPERIMENT_ID])
def test_export(experiment_id, is_async, monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "dummy-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dummy-token")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", str(is_async))
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT", "3")

    mlflow.tracing.set_destination(Databricks(experiment_id=experiment_id))

    response = mock.MagicMock()
    response.status_code = 200
    response.text = "{}"

    with mock.patch(
        "mlflow.tracing.export.databricks.http_request", return_value=response
    ) as mock_http:
        _predict("hello")

        if is_async:
            _flush_async_logging()

    mock_http.assert_called_once()
    call_args = mock_http.call_args
    assert call_args.kwargs["host_creds"] is not None
    assert call_args.kwargs["endpoint"] == "/api/2.0/tracing/traces"
    assert call_args.kwargs["method"] == "POST"
    assert call_args.kwargs["retry_timeout_seconds"] == (3 if is_async else 0)

    trace = call_args.kwargs["json"]
    trace_id = trace["info"]["trace_id"]
    assert trace_id is not None
    trace_id_b64 = base64.b64encode(int(trace_id).to_bytes(16, "big", signed=False)).decode("utf-8")
    assert trace == {
        "info": {
            "trace_id": trace_id,
            "trace_location": {
                "mlflow_experiment": {
                    "experiment_id": experiment_id or _get_experiment_id(),
                },
                "type": "MLFLOW_EXPERIMENT",
            },
            "request_preview": '{"x": "hello"}',
            "response_preview": '"hello!"',
            "request_time": mock.ANY,
            "execution_duration": mock.ANY,
            "state": "OK",
            "trace_metadata": {
                "mlflow.trace_schema.version": "2",
            },
            "tags": {
                "foo": "bar",
            },
        },
        "data": {
            "spans": [
                {
                    "trace_id": trace_id_b64,
                    "span_id": mock.ANY,
                    "trace_state": "",
                    "parent_span_id": "",
                    "name": "_predict",
                    "start_time_unix_nano": mock.ANY,
                    "end_time_unix_nano": mock.ANY,
                    "attributes": {
                        "mlflow.spanFunctionName": '"_predict"',
                        "mlflow.spanInputs": '{"x": "hello"}',
                        "mlflow.spanOutputs": '"hello!"',
                        "mlflow.spanType": '"UNKNOWN"',
                        "mlflow.traceRequestId": f'"{trace_id}"',
                    },
                    "status": {
                        "code": "STATUS_CODE_OK",
                        "message": "",
                    },
                },
                {
                    "trace_id": trace_id_b64,
                    "span_id": mock.ANY,
                    "trace_state": "",
                    "parent_span_id": mock.ANY,
                    "name": "child",
                    "start_time_unix_nano": mock.ANY,
                    "end_time_unix_nano": mock.ANY,
                    "attributes": {
                        "mlflow.spanInputs": '"dummy"',
                        "mlflow.spanType": '"UNKNOWN"',
                        "mlflow.traceRequestId": f'"{trace_id}"',
                    },
                    "events": [
                        {
                            "name": "child_event",
                            "time_unix_nano": mock.ANY,
                            "attributes": {"attr1": "val1"},
                        }
                    ],
                    "status": {
                        "code": "STATUS_CODE_OK",
                        "message": "",
                    },
                },
            ]
        },
    }

    # Last active trace ID should be set
    assert mlflow.get_last_active_trace_id() == trace_id


@pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
def test_export_catch_failure(is_async, monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "dummy-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dummy-token")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", str(is_async))
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT", "3")

    mlflow.tracing.set_destination(Databricks(experiment_id=_EXPERIMENT_ID))

    response = mock.MagicMock()
    response.status_code = 500
    response.text = "Failed to export trace"

    with (
        mock.patch(
            "mlflow.tracing.export.databricks.http_request", return_value=response
        ) as mock_http,
        mock.patch("mlflow.tracing.export.databricks._logger") as mock_logger,
    ):
        _predict("hello")

        if is_async:
            _flush_async_logging()

    mock_http.assert_called_once()
    mock_logger.warning.assert_called_once()


@pytest.mark.skipif(os.name == "nt", reason="Flaky on Windows")
def test_async_bulk_export(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "dummy-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dummy-token")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "True")
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT", "3")
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT", "3")

    mlflow.tracing.set_destination(Databricks(experiment_id=0))

    def _mock_http(*args, **kwargs):
        # Simulate a slow response
        time.sleep(0.1)
        response = mock.MagicMock()
        response.status_code = 200
        response.text = "{}"
        return response

    with mock.patch(
        "mlflow.tracing.export.databricks.http_request", side_effect=_mock_http
    ) as mock_http:
        # Log many traces
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            for _ in range(1000):
                executor.submit(_predict, "hello")

        # Trace logging should not block the main thread
        assert time.time() - start_time < 5

        _flush_async_logging()

    assert mock_http.call_count == 1000

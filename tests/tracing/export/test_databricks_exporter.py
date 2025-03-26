import base64
from unittest import mock

import httpx
import pytest

import mlflow
from mlflow.entities.span_event import SpanEvent
from mlflow.tracing.destination import Databricks

_EXPERIMENT_ID = "dummy-experiment-id"


@mlflow.trace
def _predict(x: str) -> str:
    with mlflow.start_span(name="child") as child_span:
        child_span.set_inputs("dummy")
        child_span.add_event(SpanEvent(name="child_event", attributes={"attr1": "val1"}))
    mlflow.update_current_trace(tags={"foo": "bar"})
    return x + "!"


@pytest.mark.parametrize("experiment_id", [None, _EXPERIMENT_ID])
def test_export(experiment_id, monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "dummy-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dummy-token")

    mlflow.tracing.set_destination(Databricks(experiment_id=experiment_id))

    response = mock.MagicMock()
    response.status_code = 200
    response.text = "{}"

    with mock.patch("httpx.Client.post", return_value=response) as mock_http:
        _predict("hello")

    mock_http.assert_called_once()
    call_args = mock_http.call_args
    assert call_args.kwargs["url"] == "dummy-host/api/2.0/tracing/traces"
    assert call_args.kwargs["timeout"] == 10
    assert call_args.kwargs["headers"] == {"Authorization": "Bearer dummy-token"}

    trace = call_args.kwargs["json"]
    trace_id = trace["info"]["trace_id"]
    assert trace_id is not None
    trace_id_b64 = base64.b64encode(int(trace_id).to_bytes(16, "big", signed=False)).decode("utf-8")
    assert trace == {
        "info": {
            "trace_id": trace_id,
            "trace_location": {
                "mlflow_experiment": {
                    "experiment_id": experiment_id or "0",
                },
                "type": "MLFLOW_EXPERIMENT",
            },
            "request": '{"x": "hello"}',
            "response": '"hello!"',
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


def test_export_catch_failure(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "dummy-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dummy-token")

    mlflow.tracing.set_destination(Databricks(experiment_id=_EXPERIMENT_ID))

    response = mock.MagicMock()
    response.status_code = 500
    response.text = "Failed to export trace"

    with (
        mock.patch("httpx.Client.post", return_value=response) as mock_http,
        mock.patch("mlflow.tracing.export.databricks._logger") as mock_logger,
    ):
        _predict("hello")

    mock_http.assert_called_once()
    mock_logger.warning.assert_called_once()


def test_export_override_configs(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "dummy-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dummy-token")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "20")
    monkeypatch.setenv("MLFLOW_DATABRICKS_TRACE_EXPORT_MAX_CONNECTIONS", "10")
    monkeypatch.setenv("MLFLOW_DATABRICKS_TRACE_EXPORT_KEEP_ALIVE", "5")

    mlflow.tracing.set_destination(Databricks(experiment_id=_EXPERIMENT_ID))

    response = mock.MagicMock()
    response.status_code = 200
    response.text = "{}"

    mock_http = mock.MagicMock()
    mock_http.post.return_value = response

    with mock.patch("httpx.Client", return_value=mock_http) as mock_httpx_init:
        _predict("hello")

    mock_http.post.assert_called_once()
    mock_httpx_init.assert_called_once_with(
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
    )

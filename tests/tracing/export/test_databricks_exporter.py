import base64
import json
from unittest import mock

import pytest

import mlflow
from mlflow.entities.span_event import SpanEvent
from mlflow.pyfunc.context import Context, set_prediction_context
from mlflow.tracing.destination import Databricks
from mlflow.tracing.export.inference_table import _TRACE_BUFFER

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

    with mock.patch(
        "mlflow.tracing.export.databricks.http_request", return_value=response
    ) as mock_http:
        _predict("hello")

    mock_http.assert_called_once()
    call_args = mock_http.call_args
    assert call_args.kwargs["host_creds"] is not None
    assert call_args.kwargs["endpoint"] == "/api/2.0/tracing/traces"
    assert call_args.kwargs["method"] == "POST"

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

    # By default, trace should not be written to the inference server
    assert len(_TRACE_BUFFER) == 0


def test_export_catch_failure(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "dummy-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dummy-token")

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

    mock_http.assert_called_once()
    mock_logger.warning.assert_called_once()


@pytest.mark.parametrize("is_in_serving", [True, False, None])
@pytest.mark.parametrize("is_trace_enabled", [True, False, None])
@pytest.mark.parametrize("is_dual_write_enabled", [True, False, None])
def test_export_dual_write_in_model_serving(
    is_in_serving, is_trace_enabled, is_dual_write_enabled, monkeypatch
):
    monkeypatch.setenv("DATABRICKS_HOST", "dummy-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dummy-token")

    if is_in_serving is not None:
        monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", str(is_in_serving))
    if is_trace_enabled is not None:
        monkeypatch.setenv("ENABLE_MLFLOW_TRACING", str(is_trace_enabled))
    if is_dual_write_enabled is not None:
        monkeypatch.setenv("TODO_UPDATE_ENV_VAR_NAME", str(is_dual_write_enabled))

    mlflow.tracing.set_destination(Databricks(experiment_id=_EXPERIMENT_ID))

    response = mock.MagicMock()
    response.status_code = 200
    response.text = "{}"

    databricks_request_id = "databricks-request-id"

    with mock.patch(
        "mlflow.tracing.export.databricks.http_request", return_value=response
    ) as mock_http:
        with set_prediction_context(Context(request_id=databricks_request_id)):
            _predict("hello")

    mock_http.assert_called_once()
    call_args = mock_http.call_args
    assert call_args.kwargs["endpoint"] == "/api/2.0/tracing/traces"

    trace = call_args.kwargs["json"]
    trace_id = trace["info"]["trace_id"]
    assert trace_id is not None
    # The trace_id sent to the trace server should be UUID, not the Databricks request ID
    assert trace_id != databricks_request_id

    # Dual write should only happen when all environment variables are set to True
    expect_dual_write = int(
        (is_in_serving is True) and (is_trace_enabled is True) and (is_dual_write_enabled is True)
    )
    if expect_dual_write:
        assert len(_TRACE_BUFFER) == 1
        trace = _TRACE_BUFFER.get(databricks_request_id)
        assert trace is not None
        assert trace["info"]["request_id"] == databricks_request_id
        for span in trace["data"]["spans"]:
            assert json.loads(span["attributes"]["mlflow.traceRequestId"]) == databricks_request_id
    else:
        assert len(_TRACE_BUFFER) == 0

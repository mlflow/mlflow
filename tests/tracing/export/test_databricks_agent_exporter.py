import base64
import json
from dataclasses import dataclass
from unittest import mock

import mlflow
from mlflow.entities.span_event import SpanEvent
from mlflow.tracing.destination import TraceDestination


@dataclass
class DatabricksAgentMonitoring(TraceDestination):
    databricks_monitor_id: str

    @property
    def type(self):
        return "databricks_agent_monitoring"

    def __init__(self, *, experiment_id: str):
        self.experiment_id = experiment_id
        self.databricks_monitor_id = "dummy-monitor-id"


@mlflow.trace
def _predict(x: str) -> str:
    with mlflow.start_span(name="child") as child_span:
        child_span.set_inputs("dummy")
        child_span.add_event(SpanEvent(name="child_event", attributes={"attr1": "val1"}))
    mlflow.update_current_trace(tags={"foo": "bar"})
    return x + "!"


@mock.patch("mlflow.deployments.get_deploy_client")
def test_export_legacy(mock_get_deploy_client):
    mock_deploy_client = mock.MagicMock()
    mock_get_deploy_client.return_value = mock_deploy_client

    mlflow.tracing.set_destination(
        destination=DatabricksAgentMonitoring(experiment_id="dummy-experiment-id")
    )

    _predict("hello")

    mock_deploy_client.predict.assert_called_once()
    call_args = mock_deploy_client.predict.call_args
    assert call_args.kwargs["endpoint"] == "dummy-monitor-id"
    trace = json.loads(call_args.kwargs["inputs"]["inputs"][0])
    assert trace["info"]["request_id"] is not None


def test_export_v3(monkeypatch):
    monkeypatch.setenv("AGENT_EVAL_TRACE_SERVER_ENABLED", "true")

    mlflow.tracing.set_destination(
        destination=DatabricksAgentMonitoring(experiment_id="dummy-experiment-id")
    )

    response = mock.MagicMock()
    response.status_code = 200
    response.text = "{}"

    with mock.patch(
        "mlflow.tracing.export.databricks_agent.http_request", return_value=response
    ) as mock_http:
        _predict("hello")

    mock_http.assert_called_once()
    call_args = mock_http.call_args
    assert call_args.kwargs["host_creds"] is not None
    assert call_args.kwargs["endpoint"] == "/api/2.0/tracing/traces"
    assert call_args.kwargs["method"] == "POST"

    trace_id = json.loads(call_args.kwargs["json"])["trace"]["info"]["trace_id"]
    assert trace_id is not None
    trace_id_b64 = base64.b64encode(int(trace_id).to_bytes(16, "big", signed=False)).decode("utf-8")
    assert json.loads(call_args.kwargs["json"]) == {
        "trace": {
            "info": {
                "trace_id": trace_id,
                "trace_location": {
                    "mlflow_experiment": {
                        "experiment_id": "dummy-experiment-id",
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
    }

import json
from dataclasses import dataclass
from unittest import mock

import mlflow
from mlflow.tracing.destination import TraceDestination


@mock.patch("mlflow.tracing.export.databricks_agent_legacy.get_deploy_client")
def test_export(mock_get_deploy_client):
    @dataclass
    class DatabricksAgentMonitoring(TraceDestination):
        databricks_monitor_id: str

        @property
        def type(self):
            return "databricks_agent_monitoring"

    mock_deploy_client = mock.MagicMock()
    mock_get_deploy_client.return_value = mock_deploy_client

    mlflow.tracing.set_destination(destination=DatabricksAgentMonitoring("dummy-model-endpoint"))

    with mlflow.start_span(name="root") as span:
        with mlflow.start_span(name="child") as child_span:
            child_span.set_inputs("dummy")

        otel_trace_id = span._span.context.trace_id

    mock_deploy_client.predict.assert_called_once()
    call_args = mock_deploy_client.predict.call_args
    assert call_args.kwargs["endpoint"] == "dummy-model-endpoint"
    trace = json.loads(call_args.kwargs["inputs"]["inputs"][0])
    assert trace["info"]["request_id"] == str(otel_trace_id)

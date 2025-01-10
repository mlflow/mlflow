import json
from unittest import mock

import mlflow


@mock.patch("mlflow.deployments.get_deploy_client")
def test_export(mock_get_deploy_client):
    mock_deploy_client = mock.MagicMock()
    mock_get_deploy_client.return_value = mock_deploy_client

    mlflow.tracing.set_monitoring_destination(destination="dummy-model-endpoint")

    with mlflow.start_span(name="root"):
        with mlflow.start_span(name="child") as child_span:
            child_span.set_inputs("dummy")

    mock_deploy_client.predict.assert_called_once()
    call_args = mock_deploy_client.predict.call_args
    assert call_args.kwargs["endpoint"] == "dummy-model-endpoint"
    trace = json.loads(call_args.kwargs["inputs"]["inputs"][0])
    assert trace["info"]["request_id"].startswith("tr-")

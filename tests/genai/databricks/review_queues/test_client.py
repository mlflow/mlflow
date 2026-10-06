from unittest import mock

import pytest
import requests

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.databricks.review_queues import _client


def _response(status_code: int, text: str) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = text.encode()
    return response


def test_call_rejects_non_databricks_tracking_uri():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    with pytest.raises(MlflowException, match="requires a Databricks tracking URI"):
        _client.call("GET", "experiments/1/reviewQueues")


def test_call_sends_request_and_drops_none_params():
    mlflow.set_tracking_uri("databricks")
    with (
        mock.patch.object(_client, "get_databricks_host_creds") as mock_creds,
        mock.patch.object(
            _client, "http_request", return_value=_response(200, '{"ok": true}')
        ) as mock_request,
    ):
        resp = _client.call(
            "POST",
            "experiments/1/reviewQueues",
            json={"display_name": "q"},
            params={"review_queue_id": None, "page_size": 10},
        )

    assert resp == {"ok": True}
    mock_creds.assert_called_once_with("databricks")
    mock_request.assert_called_once_with(
        host_creds=mock_creds.return_value,
        endpoint="/api/2.0/managed-evals/experiments/1/reviewQueues",
        method="POST",
        json={"display_name": "q"},
        params={"page_size": 10},
    )


def test_call_omits_json_and_empty_params():
    mlflow.set_tracking_uri("databricks")
    with (
        mock.patch.object(_client, "get_databricks_host_creds") as mock_creds,
        mock.patch.object(_client, "http_request", return_value=_response(200, "")) as mock_request,
    ):
        resp = _client.call("DELETE", "experiments/1/reviewQueues/q1", params={"x": None})

    assert resp == {}
    mock_request.assert_called_once_with(
        host_creds=mock_creds.return_value,
        endpoint="/api/2.0/managed-evals/experiments/1/reviewQueues/q1",
        method="DELETE",
    )


def test_call_raises_on_error_response():
    mlflow.set_tracking_uri("databricks")
    body = '{"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "Queue not found"}'
    with (
        mock.patch.object(_client, "get_databricks_host_creds"),
        mock.patch.object(_client, "http_request", return_value=_response(404, body)) as mock_req,
    ):
        with pytest.raises(MlflowException, match="Queue not found"):
            _client.call("GET", "experiments/1/reviewQueues/missing")
    mock_req.assert_called_once()

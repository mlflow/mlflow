import json
from unittest.mock import Mock, call

import pytest
from requests import Response

from mlflow import MlflowClient
from mlflow.entities import IssueResource
from mlflow.exceptions import MlflowNotImplementedException, RestException
from mlflow.store.tracking.databricks_rest_store import DatabricksTracingRestStore
from mlflow.store.tracking.rest_store import RestStore
from mlflow.utils.rest_utils import MlflowHostCreds

ISSUE_ID = "iss-1234567890abcdef1234567890abcdef"
COLLECTION = "/api/2.0/mlflow/experiments/1/issues"
RESOURCE_NAME = f"experiments/1/issues/{ISSUE_ID}"
OPERATIONS = [
    ("create_issue", {"experiment_id": "1", "title": "Latency", "severity": "HIGH"}),
    ("get_issue", {"experiment_id": "1", "issue_id": ISSUE_ID}),
    ("update_issue", {"experiment_id": "1", "issue_id": ISSUE_ID, "status": "RESOLVED"}),
    ("search_issues", {"experiment_id": "1"}),
    ("delete_issue", {"experiment_id": "1", "issue_id": ISSUE_ID}),
]


def _response(body=None, status_code=200):
    response = Response()
    response.status_code = status_code
    response._content = json.dumps(body).encode() if body is not None else b""
    return response


@pytest.fixture(params=[RestStore, DatabricksTracingRestStore], ids=["oss", "databricks"])
def issue_sdk(request, monkeypatch):
    tracking_uri = "databricks" if request.param is DatabricksTracingRestStore else "http://issues"
    credentials = MlflowHostCreds("https://issues", token="test-token")
    store = request.param(lambda: credentials)

    def get_store(uri):
        assert uri == tracking_uri
        return store

    http_request = Mock()
    monkeypatch.setattr("mlflow.tracing.issue_client._get_store", get_store)
    monkeypatch.setattr("mlflow.tracing.issue_client.http_request", http_request)
    return MlflowClient(tracking_uri=tracking_uri), http_request, credentials


@pytest.mark.parametrize("delete_status", [200, 204])
def test_public_issue_sdk_requests_and_responses(issue_sdk, delete_status):
    client, http_request, credentials = issue_sdk
    payload = {
        "name": RESOURCE_NAME,
        "issue_id": ISSUE_ID,
        "title": "Latency",
        "status": "PENDING",
        "severity": "HIGH",
        "description": "Slow requests",
        "categories": ["latency"],
        "tags": {"team": "api"},
        "metadata": {"count": "1", "remove.me": "old"},
        "source_type": "MANUAL",
        "last_seen_time": "2026-10-22T12:34:56Z",
    }
    updated_payload = {**payload, "status": "RESOLVED", "metadata": {"count": "2"}}
    http_request.side_effect = [
        _response(payload),
        _response(payload),
        _response(updated_payload),
        _response({"issues": [payload], "next_page_token": "next-page", "total_size": 2}),
        _response({"issues": [updated_payload], "total_size": 2}),
        _response({} if delete_status == 200 else None, delete_status),
    ]
    created = client.create_issue(
        "1",
        "Latency",
        severity="HIGH",
        description="Slow requests",
        categories=["latency"],
        tags={"team": "api"},
        metadata={"count": "1", "remove.me": "old"},
        last_seen_time=payload["last_seen_time"],
        issue_id=ISSUE_ID,
    )
    assert isinstance(created, IssueResource)
    assert created.name == RESOURCE_NAME
    assert client.get_issue("1", ISSUE_ID) == created
    updated = client.update_issue(
        "1",
        ISSUE_ID,
        status="RESOLVED",
        description=None,
        categories=[],
        tags=None,
        last_seen_time=None,
        metadata={"count": "2"},
        delete_metadata_keys=["remove.me"],
    )
    assert updated.status == "RESOLVED"
    assert updated.metadata == {"count": "2"}
    filters = {
        "query": "LATENCY",
        "statuses": ["PENDING", "RESOLVED"],
        "severities": ["HIGH"],
        "categories": ["latency"],
        "tag_filters": [{"key": "team", "values": ["api"]}],
        "sort_key": "CREATE_TIME",
        "sort_order": "ASCENDING",
        "page_size": 1,
    }
    first = client.search_issues("1", **filters)
    assert first == [created]
    assert first.total_count == 2
    assert first.token == "next-page"
    second = client.search_issues("1", **filters, page_token=first.token, view="FULL")
    assert second == [updated]
    assert second.total_count == 2
    assert second.token is None
    assert client.delete_issue("1", ISSUE_ID, force=True, sql_warehouse_id="warehouse") is None

    assert http_request.call_args_list == [
        call(
            host_creds=credentials,
            endpoint=COLLECTION,
            method="POST",
            params={"issue_id": ISSUE_ID},
            json={
                "title": "Latency",
                "severity": "HIGH",
                "description": "Slow requests",
                "status": "PENDING",
                "categories": ["latency"],
                "tags": {"team": "api"},
                "last_seen_time": payload["last_seen_time"],
                "source_type": "MANUAL",
                "metadata": {"count": "1", "remove.me": "old"},
            },
        ),
        call(
            host_creds=credentials,
            endpoint=f"{COLLECTION}/{ISSUE_ID}",
            method="GET",
            params=None,
            json=None,
        ),
        call(
            host_creds=credentials,
            endpoint=f"{COLLECTION}/{ISSUE_ID}",
            method="PATCH",
            params={
                "update_mask": "description,status,categories,tags,last_seen_time,"
                "metadata.`count`,metadata.`remove.me`"
            },
            json={
                "name": RESOURCE_NAME,
                "status": "RESOLVED",
                "categories": [],
                "metadata": {"count": "2"},
            },
        ),
        call(
            host_creds=credentials,
            endpoint=f"{COLLECTION}:search",
            method="POST",
            params=None,
            json={"parent": "experiments/1", **filters},
        ),
        call(
            host_creds=credentials,
            endpoint=f"{COLLECTION}:search",
            method="POST",
            params=None,
            json={"parent": "experiments/1", **filters, "page_token": "next-page", "view": "FULL"},
        ),
        call(
            host_creds=credentials,
            endpoint=f"{COLLECTION}/{ISSUE_ID}",
            method="DELETE",
            params={"force": "true", "sql_warehouse_id": "warehouse"},
            json=None,
        ),
    ]


@pytest.mark.parametrize(("method", "kwargs"), OPERATIONS)
def test_public_issue_sdk_preserves_server_errors(issue_sdk, method, kwargs):
    client, http_request, _ = issue_sdk
    error = {"error_code": "PERMISSION_DENIED", "message": "Experiment access denied"}
    http_request.return_value = _response(error, 403)

    with pytest.raises(RestException, match="Experiment access denied") as caught:
        getattr(client, method)(**kwargs)

    assert caught.value.error_code == "PERMISSION_DENIED"
    assert caught.value.json == error
    http_request.assert_called_once()


@pytest.mark.parametrize(("method", "kwargs"), OPERATIONS)
def test_public_issue_sdk_rejects_local_tracking_stores(monkeypatch, method, kwargs):
    client = MlflowClient(tracking_uri="http://issues")
    http_request = Mock()
    monkeypatch.setattr("mlflow.tracing.issue_client._get_store", lambda uri: object())
    monkeypatch.setattr("mlflow.tracing.issue_client.http_request", http_request)

    with pytest.raises(
        MlflowNotImplementedException, match="Local tracking stores are not supported"
    ):
        getattr(client, method)(**kwargs)

    http_request.assert_not_called()

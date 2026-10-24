import json
from unittest.mock import Mock, call

import pytest
from requests import Response

import mlflow
from mlflow.entities import IssueCategory, IssueResource, IssueSeverity, IssueStatus
from mlflow.exceptions import MlflowException, MlflowNotImplementedException, RestException
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
    monkeypatch.setattr("mlflow.tracking._tracking_service.utils._tracking_uri", tracking_uri)
    monkeypatch.setattr("mlflow.genai.issues._get_store", get_store)
    monkeypatch.setattr("mlflow.genai.issues.http_request", http_request)
    return http_request, credentials


@pytest.mark.parametrize("delete_status", [200, 204])
def test_public_issue_sdk_requests_and_responses(issue_sdk, delete_status, monkeypatch):
    http_request, credentials = issue_sdk
    payload = {
        "name": RESOURCE_NAME,
        "issue_id": ISSUE_ID,
        "title": "Latency",
        "status": "PENDING",
        "severity": "HIGH",
        "description": "Slow requests",
        "categories": ["latency", "custom-category"],
        "tags": {"team": "api"},
        "metadata": {"count": "1", "remove.me": "old"},
        "source_type": "MANUAL",
        "last_seen_time": "2026-10-22T12:34:56Z",
    }
    updated_payload = {
        **payload,
        "status": "RESOLVED",
        "severity": "MEDIUM",
        "categories": [],
        "metadata": {"count": "2"},
    }
    http_request.side_effect = [
        _response(payload),
        _response(payload),
        _response(updated_payload),
        _response({"issues": [payload], "next_page_token": "next-page", "total_size": 2}),
        _response({"issues": [updated_payload], "total_size": 2}),
        _response({} if delete_status == 200 else None, delete_status),
    ]
    created = mlflow.genai.create_issue(
        "1",
        "Latency",
        severity=IssueSeverity.HIGH,
        description="Slow requests",
        categories=[IssueCategory.LATENCY, "custom-category"],
        tags={"team": "api"},
        metadata={"count": "1", "remove.me": "old"},
        last_seen_time=payload["last_seen_time"],
        issue_id=ISSUE_ID,
    )
    assert isinstance(created, IssueResource)
    assert created.name == RESOURCE_NAME
    assert created.severity is IssueSeverity.HIGH
    assert created.categories == [IssueCategory.LATENCY, "custom-category"]
    assert created.categories[0] is IssueCategory.LATENCY
    assert mlflow.genai.get_issue("1", ISSUE_ID) == created
    updated = mlflow.genai.update_issue(
        "1",
        ISSUE_ID,
        status=IssueStatus.RESOLVED,
        severity=IssueSeverity.MEDIUM,
        description=None,
        categories=[],
        tags=None,
        last_seen_time=None,
        metadata={"count": "2"},
        delete_metadata_keys=["remove.me"],
    )
    assert updated.status is IssueStatus.RESOLVED
    assert updated.severity is IssueSeverity.MEDIUM
    assert updated.metadata == {"count": "2"}
    filters = {
        "query": "LATENCY",
        "statuses": ["PENDING", "RESOLVED"],
        "severities": ["HIGH"],
        "categories": ["latency", "custom-category"],
        "tag_filters": [{"key": "team", "values": ["api"]}],
        "sort_key": "CREATE_TIME",
        "sort_order": "ASCENDING",
        "page_size": 1,
    }
    filter_string = (
        "title ILIKE '%LATENCY%' AND status IN ('PENDING', 'RESOLVED') "
        "AND severity = 'HIGH' AND category IN ('latency', 'custom-category') "
        "AND tags.team = 'api'"
    )
    pagination = {"sort_key": "CREATE_TIME", "sort_order": "ASCENDING", "page_size": 1}
    first = mlflow.genai.search_issues("1", filter_string=filter_string, **pagination)
    assert first == [created]
    assert first.total_count == 2
    assert first.token == "next-page"
    second = mlflow.genai.search_issues(
        "1", filter_string=filter_string, **pagination, page_token=first.token, view="FULL"
    )
    assert second == [updated]
    assert second.total_count == 2
    assert second.token is None
    monkeypatch.setenv("MLFLOW_TRACING_SQL_WAREHOUSE_ID", "warehouse")
    assert mlflow.genai.delete_issue("1", ISSUE_ID, force=True) is None

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
                "categories": ["latency", "custom-category"],
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
                "update_mask": "description,status,severity,categories,tags,last_seen_time,"
                "metadata.`count`,metadata.`remove.me`"
            },
            json={
                "name": RESOURCE_NAME,
                "status": "RESOLVED",
                "severity": "MEDIUM",
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
    http_request, _ = issue_sdk
    error = {"error_code": "PERMISSION_DENIED", "message": "Experiment access denied"}
    http_request.return_value = _response(error, 403)

    with pytest.raises(RestException, match="Experiment access denied") as caught:
        getattr(mlflow.genai, method)(**kwargs)

    assert caught.value.error_code == "PERMISSION_DENIED"
    assert caught.value.json == error
    http_request.assert_called_once()


@pytest.mark.parametrize(("method", "kwargs"), OPERATIONS)
def test_public_issue_sdk_rejects_local_tracking_stores(monkeypatch, method, kwargs):
    http_request = Mock()
    monkeypatch.setattr("mlflow.genai.issues._get_store", lambda uri: object())
    monkeypatch.setattr("mlflow.genai.issues.http_request", http_request)

    with pytest.raises(
        MlflowNotImplementedException, match="Local tracking stores are not supported"
    ):
        getattr(mlflow.genai, method)(**kwargs)

    http_request.assert_not_called()


def test_issue_fluent_api_uses_current_tracking_uri(monkeypatch):
    credentials = [
        MlflowHostCreds("https://first-workspace", token="first-token"),
        MlflowHostCreds("https://second-workspace", token="second-token"),
    ]
    stores = {
        uri: RestStore(lambda creds=creds: creds)
        for uri, creds in zip(["http://first-workspace", "http://second-workspace"], credentials)
    }
    get_store = Mock(side_effect=stores.__getitem__)
    http_request = Mock(return_value=_response({"issues": []}))
    monkeypatch.setattr("mlflow.genai.issues._get_store", get_store)
    monkeypatch.setattr("mlflow.genai.issues.http_request", http_request)
    monkeypatch.setattr("mlflow.tracking._tracking_service.utils._tracking_uri", None)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    for uri in stores:
        mlflow.set_tracking_uri(uri)
        assert mlflow.genai.search_issues("1") == []

    assert get_store.call_args_list == [call(uri) for uri in stores]
    assert [request.kwargs["host_creds"] for request in http_request.call_args_list] == credentials


@pytest.mark.parametrize("warehouse_id", [None, "warehouse"])
@pytest.mark.parametrize("force", [False, True])
def test_delete_issue_reads_warehouse_from_environment(issue_sdk, monkeypatch, warehouse_id, force):
    http_request, _ = issue_sdk
    http_request.return_value = _response(status_code=204)
    if warehouse_id is None:
        monkeypatch.delenv("MLFLOW_TRACING_SQL_WAREHOUSE_ID", raising=False)
    else:
        monkeypatch.setenv("MLFLOW_TRACING_SQL_WAREHOUSE_ID", warehouse_id)

    mlflow.genai.delete_issue("1", ISSUE_ID, force=force)

    expected_params = {"force": str(force).lower()}
    if force and warehouse_id:
        expected_params["sql_warehouse_id"] = warehouse_id
    assert http_request.call_args.kwargs["params"] == expected_params


def test_delete_issue_reads_warehouse_environment_for_each_call(issue_sdk, monkeypatch):
    http_request, _ = issue_sdk
    http_request.return_value = _response(status_code=204)

    for warehouse_id in ["first-warehouse", "second-warehouse"]:
        monkeypatch.setenv("MLFLOW_TRACING_SQL_WAREHOUSE_ID", warehouse_id)
        mlflow.genai.delete_issue("1", ISSUE_ID, force=True)

    assert [request.kwargs["params"] for request in http_request.call_args_list] == [
        {"force": "true", "sql_warehouse_id": "first-warehouse"},
        {"force": "true", "sql_warehouse_id": "second-warehouse"},
    ]


@pytest.mark.parametrize("severity", [IssueSeverity.NOT_AN_ISSUE, "not_an_issue", "CRITICAL"])
@pytest.mark.parametrize("operation", ["create", "update"])
def test_issue_severity_rejects_values_not_supported_by_rest(issue_sdk, severity, operation):
    http_request, _ = issue_sdk
    kwargs = {"title": "Latency"} if operation == "create" else {"issue_id": ISSUE_ID}
    with pytest.raises(MlflowException, match="(?i)severity") as caught:
        getattr(mlflow.genai, f"{operation}_issue")("1", severity=severity, **kwargs)

    assert caught.value.error_code == "INVALID_PARAMETER_VALUE"
    http_request.assert_not_called()


@pytest.mark.parametrize(
    "field_value", [{"status": "INVALID"}, {"categories": "latency"}, {"categories": [12]}]
)
@pytest.mark.parametrize("operation", ["create", "update"])
def test_issue_rejects_invalid_fields_before_request(issue_sdk, field_value, operation):
    http_request, _ = issue_sdk
    kwargs = (
        {"title": "Latency", "severity": IssueSeverity.HIGH}
        if operation == "create"
        else {"issue_id": ISSUE_ID}
    )
    with pytest.raises(MlflowException, match=f"(?i){next(iter(field_value))}") as caught:
        getattr(mlflow.genai, f"{operation}_issue")("1", **kwargs, **field_value)

    assert caught.value.error_code == "INVALID_PARAMETER_VALUE"
    http_request.assert_not_called()

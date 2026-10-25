import json
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest
from requests import Response

from mlflow.entities import Issue, IssueCategory, IssueSeverity, IssueStatus
from mlflow.exceptions import MlflowException, RestException
from mlflow.store.tracking.databricks_rest_store import DatabricksTracingRestStore
from mlflow.utils.rest_utils import MlflowHostCreds

ISSUE_ID = "iss-1234567890abcdef1234567890abcdef"
COLLECTION = "/api/2.0/mlflow/experiments/1/issues"
RESOURCE_NAME = f"experiments/1/issues/{ISSUE_ID}"


def _response(body=None, status_code=200):
    response = Response()
    response.status_code = status_code
    response._content = json.dumps(body).encode() if body is not None else b""
    return response


@pytest.fixture
def issue_payload():
    return {
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
        "create_time": "2026-10-22T12:34:00Z",
        "update_time": "2026-10-22T12:35:00Z",
        "last_seen_time": "2026-10-22T12:34:56Z",
    }


@pytest.fixture
def managed_store(monkeypatch):
    credentials = MlflowHostCreds("https://issues", token="test-token")
    http_request = Mock()
    monkeypatch.setattr("mlflow.utils.rest_utils.http_request", http_request)
    return DatabricksTracingRestStore(lambda: credentials), http_request, credentials


@pytest.mark.parametrize("status_code", [200, 201])
def test_create_issue_serializes_managed_request(managed_store, issue_payload, status_code):
    store, http_request, credentials = managed_store
    http_request.return_value = _response(issue_payload, status_code)

    issue = store.create_issue(
        experiment_id="1",
        name="Latency",
        description="Slow requests",
        status=IssueStatus.PENDING,
        severity=IssueSeverity.HIGH,
        categories=[IssueCategory.LATENCY, "custom-category"],
        tags={"team": "api"},
        last_seen_timestamp=1792672496000,
        source_type="MANUAL",
        metadata={"count": "1", "remove.me": "old"},
        issue_id=ISSUE_ID,
    )

    assert isinstance(issue, Issue)
    assert issue.name == "Latency"
    assert issue.experiment_id == "1"
    assert issue.created_timestamp == 1792672440000
    assert issue.last_updated_timestamp == 1792672500000
    assert issue.last_seen_timestamp == 1792672496000
    assert issue.tags == {"team": "api"}
    assert issue.source_type == "MANUAL"
    assert issue.severity is IssueSeverity.HIGH
    assert issue.categories == [IssueCategory.LATENCY, "custom-category"]
    assert issue.categories[0] is IssueCategory.LATENCY
    request = http_request.call_args.kwargs
    endpoint = urlsplit(request["endpoint"])
    assert endpoint.path == COLLECTION
    assert parse_qs(endpoint.query) == {"issue_id": [ISSUE_ID]}
    assert request["host_creds"] is credentials
    assert request["method"] == "POST"
    assert request["json"] == {
        "title": "Latency",
        "severity": "HIGH",
        "description": "Slow requests",
        "status": "PENDING",
        "categories": ["latency", "custom-category"],
        "tags": {"team": "api"},
        "last_seen_time": "2026-10-22T12:34:56Z",
        "source_type": "MANUAL",
        "metadata": {"count": "1", "remove.me": "old"},
    }


@pytest.mark.parametrize(
    "source_type", ["ADHOC_RUN", "SCHEDULED_RUN", "SINGLE_TRACE_ANALYSIS", "MANUAL"]
)
def test_get_issue_converts_managed_response(managed_store, issue_payload, source_type):
    store, http_request, credentials = managed_store
    http_request.return_value = _response({
        **issue_payload,
        "title": "Payment failures",
        "description": "Retries fail after a timeout.",
        "status": "RESOLVED",
        "severity": "MEDIUM",
        "create_time": "1970-01-01T01:00:00.123456789+01:00",
        "update_time": "1970-01-01T01:00:02.987654321+01:00",
        "last_seen_time": "1970-01-01T00:00:01.999999999Z",
        "categories": ["correctness", "payment-processing"],
        "source_type": source_type,
        "source_id": "source-123",
        "tags": {"team": "payments"},
        "metadata": {"recent_affected_trace_ids": '["tr-123"]'},
        "created_by": "creator",
        "last_updated_by": "editor",
    })

    issue = store.get_issue(ISSUE_ID, experiment_id="1")

    assert isinstance(issue, Issue)
    assert issue.issue_id == ISSUE_ID
    assert issue.experiment_id == "1"
    assert issue.name == "Payment failures"
    assert issue.description == "Retries fail after a timeout."
    assert issue.created_timestamp == 123
    assert issue.last_updated_timestamp == 2987
    assert issue.last_seen_timestamp == 1999
    assert issue.status is IssueStatus.RESOLVED
    assert issue.severity is IssueSeverity.MEDIUM
    assert issue.categories == [IssueCategory.CORRECTNESS, "payment-processing"]
    assert issue.categories[0] is IssueCategory.CORRECTNESS
    assert issue.source_type == source_type
    assert issue.source_id == "source-123"
    assert issue.source_run_id == (
        "source-123" if source_type in {"ADHOC_RUN", "SCHEDULED_RUN"} else None
    )
    assert issue.tags == {"team": "payments"}
    assert issue.metadata == {"recent_affected_trace_ids": '["tr-123"]'}
    assert issue.created_by == "creator"
    assert issue.last_updated_by == "editor"
    http_request.assert_called_once_with(
        host_creds=credentials,
        endpoint=f"{COLLECTION}/{ISSUE_ID}",
        method="GET",
        params={},
    )


def test_update_issue_preserves_clear_operations_in_update_mask(managed_store, issue_payload):
    store, http_request, credentials = managed_store
    http_request.return_value = _response({
        **issue_payload,
        "description": "",
        "status": "RESOLVED",
        "severity": "MEDIUM",
        "categories": [],
        "tags": {},
        "metadata": {"count": "2"},
    })

    issue = store.update_issue(
        ISSUE_ID,
        experiment_id="1",
        description="",
        status=IssueStatus.RESOLVED,
        severity=IssueSeverity.MEDIUM,
        categories=[],
        tags={},
        metadata={"count": "2"},
        delete_metadata_keys=["remove.me"],
    )

    assert issue.status is IssueStatus.RESOLVED
    assert issue.severity is IssueSeverity.MEDIUM
    assert issue.description == ""
    assert issue.categories == []
    assert issue.tags == {}
    assert issue.metadata == {"count": "2"}
    request = http_request.call_args.kwargs
    endpoint = urlsplit(request["endpoint"])
    assert endpoint.path == f"{COLLECTION}/{ISSUE_ID}"
    assert set(parse_qs(endpoint.query)["update_mask"][0].split(",")) == {
        "description",
        "status",
        "severity",
        "categories",
        "tags",
        "metadata.`count`",
        "metadata.`remove.me`",
    }
    assert request["host_creds"] is credentials
    assert request["method"] == "PATCH"
    assert request["json"] == {
        "name": RESOURCE_NAME,
        "description": "",
        "status": "RESOLVED",
        "severity": "MEDIUM",
        "metadata": {"count": "2"},
    }


def test_update_issue_omits_unchanged_fields(managed_store, issue_payload):
    store, http_request, _ = managed_store
    http_request.return_value = _response(issue_payload)

    store.update_issue(ISSUE_ID, experiment_id="1", status=IssueStatus.RESOLVED)

    request = http_request.call_args.kwargs
    assert parse_qs(urlsplit(request["endpoint"]).query) == {"update_mask": ["status"]}
    assert request["json"] == {"name": RESOURCE_NAME, "status": "RESOLVED"}


def test_update_issue_preserves_zero_timestamp(managed_store, issue_payload):
    store, http_request, _ = managed_store
    http_request.return_value = _response(issue_payload)

    store.update_issue(ISSUE_ID, experiment_id="1", last_seen_timestamp=0)

    request = http_request.call_args.kwargs
    assert parse_qs(urlsplit(request["endpoint"]).query) == {"update_mask": ["last_seen_time"]}
    assert request["json"] == {"name": RESOURCE_NAME, "last_seen_time": "1970-01-01T00:00:00Z"}


def test_search_issues_translates_filters_and_preserves_pagination(managed_store, issue_payload):
    store, http_request, credentials = managed_store
    http_request.side_effect = [
        _response({"issues": [issue_payload], "next_page_token": "next-page", "total_size": 2}),
        _response({"issues": [issue_payload], "total_size": 2}),
    ]
    kwargs = {
        "experiment_id": "1",
        "filter_string": (
            "title ILIKE '%LATENCY%' AND status IN ('PENDING', 'RESOLVED') "
            "AND severity = 'HIGH' AND category IN ('latency', 'custom-category') "
            "AND tags.team = 'api'"
        ),
        "max_results": 1,
        "order_by": ["created_timestamp ASC"],
        "include_total_count": True,
    }

    first = store.search_issues(**kwargs)
    second = store.search_issues(**kwargs, page_token=first.token, view="full")

    assert isinstance(first[0], Issue)
    assert first[0].name == "Latency"
    assert first.total_count == second.total_count == 2
    assert first.token == "next-page"
    assert second.token is None
    filters = {
        "parent": "experiments/1",
        "query": "LATENCY",
        "statuses": ["PENDING", "RESOLVED"],
        "severities": ["HIGH"],
        "categories": ["latency", "custom-category"],
        "tag_filters": [{"key": "team", "values": ["api"]}],
        "sort_key": "CREATE_TIME",
        "sort_order": "ASCENDING",
        "page_size": 1,
    }
    assert [request.kwargs for request in http_request.call_args_list] == [
        {
            "host_creds": credentials,
            "endpoint": f"{COLLECTION}:search",
            "method": "POST",
            "json": filters,
        },
        {
            "host_creds": credentials,
            "endpoint": f"{COLLECTION}:search",
            "method": "POST",
            "json": {**filters, "page_token": "next-page", "view": "FULL"},
        },
    ]


@pytest.mark.parametrize("severity", [None, "", "ISSUE_SEVERITY_UNSPECIFIED"])
def test_search_issue_basic_view_preserves_missing_optional_fields(managed_store, severity):
    store, http_request, _ = managed_store
    payload = {
        "name": RESOURCE_NAME,
        "issue_id": ISSUE_ID,
        "title": "Manual issue",
        "status": "PENDING",
        "create_time": "2026-10-22T12:34:00Z",
        "update_time": "2026-10-22T12:35:00Z",
    }
    if severity is not None:
        payload["severity"] = severity
    http_request.return_value = _response({"issues": [payload]})

    issues = store.search_issues("1", view="basic")
    [issue] = issues

    assert issues.token is None
    assert issues.total_count is None
    assert issue.description == ""
    assert issue.severity is None
    assert issue.last_seen_timestamp is None
    assert issue.source_id is None
    assert issue.source_run_id is None
    assert issue.categories == []
    assert issue.tags == {}
    assert issue.metadata == {}


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        (
            "create_issue",
            {"name": "Latency", "description": "", "severity": IssueSeverity.HIGH},
        ),
        ("get_issue", {"issue_id": ISSUE_ID}),
        ("update_issue", {"issue_id": ISSUE_ID, "status": IssueStatus.RESOLVED}),
        ("search_issues", {}),
        ("delete_issue", {"issue_id": ISSUE_ID}),
    ],
)
def test_managed_issues_preserve_server_errors(managed_store, method, kwargs):
    store, http_request, _ = managed_store
    error = {"error_code": "PERMISSION_DENIED", "message": "Experiment access denied"}
    http_request.return_value = _response(error, 403)

    with pytest.raises(RestException, match="Experiment access denied") as caught:
        getattr(store, method)(experiment_id="1", **kwargs)

    assert caught.value.error_code == "PERMISSION_DENIED"
    assert caught.value.json == error
    http_request.assert_called_once()


@pytest.mark.parametrize("severity", [IssueSeverity.NOT_AN_ISSUE, "not_an_issue", "CRITICAL"])
@pytest.mark.parametrize("operation", ["create", "update"])
def test_issue_severity_rejects_values_not_supported_by_managed_api(
    managed_store, severity, operation
):
    store, http_request, _ = managed_store
    kwargs = (
        {"name": "Latency", "description": ""} if operation == "create" else {"issue_id": ISSUE_ID}
    )

    with pytest.raises(MlflowException, match="(?i)severity") as caught:
        getattr(store, f"{operation}_issue")(experiment_id="1", severity=severity, **kwargs)

    assert caught.value.error_code == "INVALID_PARAMETER_VALUE"
    http_request.assert_not_called()


@pytest.mark.parametrize(
    "filter_string",
    ["severity = 'unknown'", "status = 'deleted'", "severity = 'NOT_AN_ISSUE'"],
)
def test_search_issues_rejects_invalid_enum_values(managed_store, filter_string):
    store, http_request, _ = managed_store

    with pytest.raises(MlflowException, match="(?i)(severity|status)") as caught:
        store.search_issues("1", filter_string=filter_string)

    assert caught.value.error_code == "INVALID_PARAMETER_VALUE"
    http_request.assert_not_called()


@pytest.mark.parametrize("status_code", [200, 204])
@pytest.mark.parametrize("warehouse_id", [None, "warehouse"])
@pytest.mark.parametrize("force", [False, True])
def test_delete_issue_uses_warehouse_environment(
    managed_store, monkeypatch, status_code, warehouse_id, force
):
    store, http_request, credentials = managed_store
    http_request.return_value = _response({} if status_code == 200 else None, status_code)
    if warehouse_id is None:
        monkeypatch.delenv("MLFLOW_TRACING_SQL_WAREHOUSE_ID", raising=False)
    else:
        monkeypatch.setenv("MLFLOW_TRACING_SQL_WAREHOUSE_ID", warehouse_id)

    assert store.delete_issue(ISSUE_ID, experiment_id="1", force=force) is None

    request = http_request.call_args.kwargs
    endpoint = urlsplit(request["endpoint"])
    expected_query = {"force": [str(force).lower()]}
    if force and warehouse_id:
        expected_query["sql_warehouse_id"] = [warehouse_id]
    assert endpoint.path == f"{COLLECTION}/{ISSUE_ID}"
    assert parse_qs(endpoint.query) == expected_query
    assert request["host_creds"] is credentials
    assert request["method"] == "DELETE"
    assert request["json"] is None


def test_delete_issue_reads_warehouse_environment_for_each_call(managed_store, monkeypatch):
    store, http_request, _ = managed_store
    http_request.return_value = _response(status_code=204)

    for warehouse_id in ["first-warehouse", "second-warehouse"]:
        monkeypatch.setenv("MLFLOW_TRACING_SQL_WAREHOUSE_ID", warehouse_id)
        store.delete_issue(ISSUE_ID, experiment_id="1", force=True)

    assert [
        parse_qs(urlsplit(request.kwargs["endpoint"]).query)
        for request in http_request.call_args_list
    ] == [
        {"force": ["true"], "sql_warehouse_id": ["first-warehouse"]},
        {"force": ["true"], "sql_warehouse_id": ["second-warehouse"]},
    ]


@pytest.mark.parametrize("force", ["false", 1, None])
def test_delete_issue_rejects_non_boolean_force(managed_store, force):
    store, http_request, _ = managed_store

    with pytest.raises(MlflowException, match="force") as caught:
        store.delete_issue(ISSUE_ID, experiment_id="1", force=force)

    assert caught.value.error_code == "INVALID_PARAMETER_VALUE"
    http_request.assert_not_called()

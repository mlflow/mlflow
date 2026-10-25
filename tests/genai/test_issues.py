from unittest.mock import Mock

import pytest

import mlflow
from mlflow.entities import IssueCategory, IssueSeverity, IssueStatus
from mlflow.exceptions import MlflowException
from mlflow.tracking._tracking_service.utils import _use_tracking_uri

ISSUE_ID = "iss-1234567890abcdef1234567890abcdef"


@pytest.fixture
def tracing_client(monkeypatch):
    client = Mock()
    monkeypatch.setattr("mlflow.genai.issues.TracingClient", Mock(return_value=client))
    return client


def test_create_issue_delegates_to_tracing_client(tracing_client):
    kwargs = {
        "experiment_id": "1",
        "name": "Latency",
        "description": "Slow requests",
        "status": IssueStatus.PENDING,
        "severity": IssueSeverity.HIGH,
        "categories": [IssueCategory.LATENCY, "custom-category"],
        "last_seen_timestamp": 1792672496000,
        "source_run_id": "source-run",
        "metadata": {"count": "1"},
        "issue_id": ISSUE_ID,
    }

    result = mlflow.genai.create_issue(**kwargs)

    assert result is tracing_client._create_issue.return_value
    tracing_client._create_issue.assert_called_once_with(**kwargs)


def test_get_issue_delegates_to_tracing_client(tracing_client):
    result = mlflow.genai.get_issue("1", ISSUE_ID)

    assert result is tracing_client._get_issue.return_value
    tracing_client._get_issue.assert_called_once_with(ISSUE_ID, experiment_id="1")


def test_update_issue_delegates_to_tracing_client(tracing_client):
    kwargs = {
        "name": "New name",
        "description": "",
        "status": IssueStatus.RESOLVED,
        "severity": IssueSeverity.MEDIUM,
        "categories": [],
        "last_seen_timestamp": 1792672496000,
        "metadata": {"count": "2"},
        "delete_metadata_keys": ["remove.me"],
    }

    result = mlflow.genai.update_issue("1", ISSUE_ID, **kwargs)

    assert result is tracing_client._update_issue.return_value
    tracing_client._update_issue.assert_called_once_with(ISSUE_ID, experiment_id="1", **kwargs)


def test_search_issues_delegates_to_tracing_client(tracing_client):
    kwargs = {
        "experiment_id": "1",
        "filter_string": "status = 'pending'",
        "max_results": 20,
        "order_by": ["created_timestamp ASC"],
        "page_token": "next-page",
        "view": "full",
    }

    result = mlflow.genai.search_issues(**kwargs)

    assert result is tracing_client._search_issues.return_value
    tracing_client._search_issues.assert_called_once_with(**kwargs)


def test_issue_api_uses_existing_local_store():
    experiment_id = mlflow.create_experiment("issue-sdk")
    with mlflow.start_run(experiment_id=experiment_id) as run:
        source_run_id = run.info.run_id
    created = mlflow.genai.create_issue(
        experiment_id,
        "Latency",
        severity=IssueSeverity.HIGH,
        description="Slow requests",
        categories=[IssueCategory.LATENCY, "custom-category"],
        source_run_id=source_run_id,
        last_seen_timestamp=0,
        metadata={"count": "1", "remove.me": "old"},
        issue_id=ISSUE_ID,
    )

    assert created.issue_id == ISSUE_ID
    assert created.source_run_id == source_run_id
    assert created.status is IssueStatus.PENDING
    assert created.severity is IssueSeverity.HIGH
    assert created.categories == [IssueCategory.LATENCY, "custom-category"]
    assert created.last_seen_timestamp == 0
    assert mlflow.genai.get_issue(experiment_id, ISSUE_ID) == created

    unchanged = mlflow.genai.update_issue(experiment_id, ISSUE_ID)
    assert unchanged.name == "Latency"
    assert unchanged.description == "Slow requests"
    assert unchanged.categories == created.categories
    assert unchanged.metadata == created.metadata
    assert unchanged.last_seen_timestamp == 0

    updated = mlflow.genai.update_issue(
        experiment_id,
        ISSUE_ID,
        status=IssueStatus.RESOLVED,
        description="",
        categories=[],
        metadata={"count": "2"},
        delete_metadata_keys=["remove.me"],
    )
    assert updated.status is IssueStatus.RESOLVED
    assert updated.description == ""
    assert not updated.categories
    assert updated.metadata == {"count": "2"}
    assert updated.last_seen_timestamp == 0
    assert mlflow.genai.get_issue(experiment_id, ISSUE_ID) == updated


def test_search_issues_filters_and_paginates_local_store():
    experiment_id = mlflow.create_experiment("issue-sdk-search")
    for name in ["A", "B"]:
        mlflow.genai.create_issue(
            experiment_id,
            name,
            severity=IssueSeverity.HIGH,
            description="Details",
            metadata={"key": "value"},
        )
    mlflow.genai.create_issue(
        experiment_id, "Resolved", severity=IssueSeverity.LOW, status=IssueStatus.RESOLVED
    )
    kwargs = {
        "filter_string": "status = 'pending'",
        "max_results": 1,
        "order_by": ["name ASC"],
        "view": "basic",
    }

    first = mlflow.genai.search_issues(experiment_id, **kwargs)
    second = mlflow.genai.search_issues(experiment_id, page_token=first.token, **kwargs)

    assert [issue.name for issue in first] == ["A"]
    assert [issue.name for issue in second] == ["B"]
    assert first.token is not None
    assert second.token is None
    assert first.total_count is None
    assert second.total_count is None
    assert first[0].description == ""
    assert first[0].metadata is None


def test_issue_api_uses_current_tracking_uri(tmp_sqlite_uri):
    experiment_id = mlflow.create_experiment("original-store")
    original = mlflow.genai.create_issue(experiment_id, "Original", severity=IssueSeverity.HIGH)

    with _use_tracking_uri(tmp_sqlite_uri):
        other_experiment_id = mlflow.create_experiment("other-store")
        assert mlflow.genai.search_issues(other_experiment_id) == []
        other = mlflow.genai.create_issue(
            other_experiment_id, "Other", severity=IssueSeverity.MEDIUM
        )
        assert mlflow.genai.get_issue(other_experiment_id, other.issue_id).name == "Other"
        assert [issue.name for issue in mlflow.genai.search_issues(other_experiment_id)] == [
            "Other"
        ]

    assert mlflow.genai.get_issue(experiment_id, original.issue_id).name == "Original"
    assert [issue.name for issue in mlflow.genai.search_issues(experiment_id)] == ["Original"]


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("create_issue", {"name": "Latency", "severity": IssueSeverity.HIGH}),
        ("get_issue", {"issue_id": ISSUE_ID}),
        ("update_issue", {"issue_id": ISSUE_ID, "status": IssueStatus.RESOLVED}),
        ("search_issues", {"filter_string": "status = 'pending'"}),
    ],
)
def test_issue_api_preserves_client_errors(tracing_client, method, kwargs):
    error = MlflowException("Experiment access denied", "PERMISSION_DENIED")
    getattr(tracing_client, f"_{method}").side_effect = error

    with pytest.raises(MlflowException, match="Experiment access denied") as caught:
        getattr(mlflow.genai, method)("1", **kwargs)

    assert caught.value is error

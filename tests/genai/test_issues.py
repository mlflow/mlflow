from unittest.mock import Mock

import pytest

import mlflow
from mlflow.entities import IssueCategory, IssueSeverity, IssueStatus
from mlflow.exceptions import MlflowException

ISSUE_ID = "iss-1234567890abcdef1234567890abcdef"


@pytest.fixture
def tracing_client(monkeypatch):
    client = Mock()
    constructor = Mock(return_value=client)
    monkeypatch.setattr("mlflow.genai.issues.TracingClient", constructor)
    return client


def test_create_issue_delegates_to_tracing_client(tracing_client):
    kwargs = {
        "experiment_id": "1",
        "name": "Latency",
        "description": "Slow requests",
        "status": IssueStatus.PENDING,
        "severity": IssueSeverity.HIGH,
        "categories": [IssueCategory.LATENCY, "custom-category"],
        "tags": {"team": "api"},
        "last_seen_timestamp": 1792672496000,
        "source_type": "MANUAL",
        "source_id": "source-123",
        "metadata": {"count": "1"},
        "issue_id": ISSUE_ID,
    }

    result = mlflow.genai.create_issue(**kwargs)

    assert result is tracing_client._create_issue.return_value
    tracing_client._create_issue.assert_called_once_with(**kwargs)


def test_create_issue_defaults(tracing_client):
    mlflow.genai.create_issue("1", "Latency", severity=IssueSeverity.HIGH)

    tracing_client._create_issue.assert_called_once_with(
        experiment_id="1",
        name="Latency",
        description="",
        status=IssueStatus.PENDING,
        severity=IssueSeverity.HIGH,
        categories=None,
        tags=None,
        last_seen_timestamp=None,
        source_type=None,
        source_id=None,
        metadata=None,
        issue_id=None,
    )


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
        "tags": {},
        "last_seen_timestamp": 1792672496000,
        "metadata": {"count": "2"},
        "delete_metadata_keys": ["remove.me"],
    }

    result = mlflow.genai.update_issue("1", ISSUE_ID, **kwargs)

    assert result is tracing_client._update_issue.return_value
    tracing_client._update_issue.assert_called_once_with(ISSUE_ID, experiment_id="1", **kwargs)


def test_update_issue_defaults_preserve_fields(tracing_client):
    mlflow.genai.update_issue("1", ISSUE_ID, status=IssueStatus.RESOLVED)

    tracing_client._update_issue.assert_called_once_with(
        ISSUE_ID,
        experiment_id="1",
        name=None,
        description=None,
        status=IssueStatus.RESOLVED,
        severity=None,
        categories=None,
        tags=None,
        last_seen_timestamp=None,
        metadata=None,
        delete_metadata_keys=None,
    )


def test_search_issues_delegates_to_tracing_client(tracing_client):
    kwargs = {
        "experiment_id": "1",
        "filter_string": "status = 'PENDING'",
        "max_results": 20,
        "order_by": ["created_timestamp ASC"],
        "page_token": "next-page",
        "view": "full",
    }

    result = mlflow.genai.search_issues(**kwargs)

    assert result is tracing_client._search_issues.return_value
    tracing_client._search_issues.assert_called_once_with(**kwargs, include_total_count=True)


def test_search_issues_defaults(tracing_client):
    mlflow.genai.search_issues("1")

    tracing_client._search_issues.assert_called_once_with(
        experiment_id="1",
        filter_string=None,
        max_results=None,
        order_by=None,
        page_token=None,
        view=None,
        include_total_count=True,
    )


@pytest.mark.parametrize("force", [False, True])
def test_delete_issue_delegates_to_tracing_client(tracing_client, force):
    assert mlflow.genai.delete_issue("1", ISSUE_ID, force=force) is None

    tracing_client._delete_issue.assert_called_once_with(ISSUE_ID, experiment_id="1", force=force)


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("create_issue", {"name": "Latency", "severity": IssueSeverity.HIGH}),
        ("get_issue", {"issue_id": ISSUE_ID}),
        ("update_issue", {"issue_id": ISSUE_ID, "status": IssueStatus.RESOLVED}),
        ("search_issues", {"filter_string": "status = 'PENDING'"}),
        ("delete_issue", {"issue_id": ISSUE_ID}),
    ],
)
def test_issue_api_preserves_client_errors(tracing_client, method, kwargs):
    error = MlflowException("Experiment access denied", "PERMISSION_DENIED")
    getattr(tracing_client, f"_{method}").side_effect = error

    with pytest.raises(MlflowException, match="Experiment access denied") as caught:
        getattr(mlflow.genai, method)("1", **kwargs)

    assert caught.value is error

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
    experiment = mlflow.set_experiment("issue-sdk-lookup")
    result = mlflow.genai.get_issue(ISSUE_ID)

    assert result is tracing_client._get_issue.return_value
    tracing_client._get_issue.assert_called_once_with(
        ISSUE_ID, experiment_id=experiment.experiment_id
    )
    tracing_client._get_issue.reset_mock()

    assert mlflow.genai.get_issue(ISSUE_ID, experiment_id="override") is result
    tracing_client._get_issue.assert_called_once_with(ISSUE_ID, experiment_id="override")


def test_update_issue_delegates_to_tracing_client(tracing_client):
    kwargs = {
        "experiment_id": "1",
        "name": "New name",
        "description": "",
        "status": IssueStatus.RESOLVED,
        "severity": IssueSeverity.MEDIUM,
        "categories": [],
        "last_seen_timestamp": 1792672496000,
        "metadata": {"count": "2"},
        "delete_metadata_keys": ["remove.me"],
    }

    result = mlflow.genai.update_issue(ISSUE_ID, **kwargs)

    assert result is tracing_client._update_issue.return_value
    tracing_client._update_issue.assert_called_once_with(ISSUE_ID, **kwargs)


def test_search_issues_delegates_to_tracing_client(tracing_client):
    kwargs = {
        "experiment_id": "1",
        "filter_string": "status = 'pending'",
        "name": "latency",
        "severities": [IssueSeverity.HIGH, IssueSeverity.MEDIUM],
        "categories": [IssueCategory.LATENCY, "custom-category"],
        "max_results": 20,
        "order_by": ["created_timestamp ASC"],
        "page_token": "next-page",
        "include_total_count": True,
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
        "Latency",
        severity=IssueSeverity.HIGH,
        description="Slow requests",
        categories=[IssueCategory.LATENCY, "custom-category"],
        source_run_id=source_run_id,
        last_seen_timestamp=0,
        metadata={"count": "1", "remove.me": "old"},
        issue_id=ISSUE_ID,
        experiment_id=experiment_id,
    )

    assert created.issue_id == ISSUE_ID
    assert created.source_run_id == source_run_id
    assert created.status is IssueStatus.PENDING
    assert created.severity is IssueSeverity.HIGH
    assert created.categories == [IssueCategory.LATENCY, "custom-category"]
    assert created.last_seen_timestamp == 0
    assert mlflow.genai.get_issue(ISSUE_ID) == created

    unchanged = mlflow.genai.update_issue(ISSUE_ID)
    assert unchanged.name == "Latency"
    assert unchanged.description == "Slow requests"
    assert unchanged.categories == created.categories
    assert unchanged.metadata == created.metadata
    assert unchanged.last_seen_timestamp == 0

    updated = mlflow.genai.update_issue(
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
    assert mlflow.genai.get_issue(ISSUE_ID) == updated


@pytest.mark.parametrize("include_total_count", [True, False])
def test_search_issues_filters_and_paginates_local_store(include_total_count):
    experiment_id = mlflow.create_experiment("issue-sdk-search")
    for name, severity, status, categories in [
        ("A latency%", IssueSeverity.HIGH, IssueStatus.PENDING, [IssueCategory.LATENCY]),
        ("B LATENCY%", IssueSeverity.MEDIUM, IssueStatus.PENDING, ["custom-category"]),
        ("Resolved latency%", IssueSeverity.HIGH, IssueStatus.RESOLVED, [IssueCategory.LATENCY]),
        ("Low latency%", IssueSeverity.LOW, IssueStatus.PENDING, [IssueCategory.LATENCY]),
        ("No category latency%", IssueSeverity.HIGH, IssueStatus.PENDING, []),
        ("Other issue", IssueSeverity.HIGH, IssueStatus.PENDING, [IssueCategory.LATENCY]),
        ("Wildcard latencyX", IssueSeverity.HIGH, IssueStatus.PENDING, [IssueCategory.LATENCY]),
    ]:
        mlflow.genai.create_issue(
            name,
            severity=severity,
            status=status,
            categories=categories,
            description="Details",
            metadata={"key": "value"},
            experiment_id=experiment_id,
        )
    kwargs = {
        "filter_string": "status = 'pending'",
        "name": "latency%",
        "severities": [IssueSeverity.HIGH, IssueSeverity.MEDIUM],
        "categories": [" LATENCY ", "CUSTOM-CATEGORY"],
        "max_results": 1,
        "order_by": ["name ASC"],
        "include_total_count": include_total_count,
        "view": "basic",
    }

    first = mlflow.genai.search_issues(experiment_id=experiment_id, **kwargs)
    second = mlflow.genai.search_issues(
        experiment_id=experiment_id, page_token=first.token, **kwargs
    )

    assert [issue.name for issue in first] == ["A latency%"]
    assert [issue.name for issue in second] == ["B LATENCY%"]
    assert first.token is not None
    assert second.token is None
    assert first.total_count == (2 if include_total_count else None)
    assert second.total_count == (2 if include_total_count else None)
    assert first[0].description == ""
    assert first[0].metadata is None


def test_issue_api_uses_current_tracking_uri(tmp_sqlite_uri):
    experiment_id = mlflow.create_experiment("original-store")
    original = mlflow.genai.create_issue(
        "Original",
        description="Original issue",
        severity=IssueSeverity.HIGH,
        experiment_id=experiment_id,
    )

    with _use_tracking_uri(tmp_sqlite_uri):
        other_experiment_id = mlflow.create_experiment("other-store")
        assert mlflow.genai.search_issues(experiment_id=other_experiment_id) == []
        other = mlflow.genai.create_issue(
            "Other",
            description="Other issue",
            severity=IssueSeverity.MEDIUM,
            experiment_id=other_experiment_id,
        )
        assert mlflow.genai.get_issue(other.issue_id).name == "Other"
        assert [
            issue.name for issue in mlflow.genai.search_issues(experiment_id=other_experiment_id)
        ] == ["Other"]

    assert mlflow.genai.get_issue(original.issue_id).name == "Original"
    assert [issue.name for issue in mlflow.genai.search_issues(experiment_id=experiment_id)] == [
        "Original"
    ]


def test_issue_api_infers_active_experiment_and_supports_explicit_override():
    first_experiment = mlflow.set_experiment("first-issue-experiment")
    first = mlflow.genai.create_issue(
        "First", description="First issue", severity=IssueSeverity.HIGH
    )
    assert first.experiment_id == first_experiment.experiment_id

    second_experiment = mlflow.set_experiment("second-issue-experiment")
    second = mlflow.genai.create_issue(
        "Second", description="Second issue", severity=IssueSeverity.LOW
    )
    assert second.experiment_id == second_experiment.experiment_id
    assert [issue.name for issue in mlflow.genai.search_issues("status = 'pending'")] == ["Second"]

    override = mlflow.genai.create_issue(
        "Override",
        description="Explicit experiment",
        severity=IssueSeverity.MEDIUM,
        experiment_id=first_experiment.experiment_id,
    )
    assert override.experiment_id == first_experiment.experiment_id
    assert [
        issue.name
        for issue in mlflow.genai.search_issues(
            experiment_id=first_experiment.experiment_id, order_by=["name ASC"]
        )
    ] == ["First", "Override"]

    assert mlflow.genai.get_issue(first.issue_id).name == "First"
    updated = mlflow.genai.update_issue(first.issue_id, name="Updated")
    assert updated.experiment_id == first_experiment.experiment_id
    assert mlflow.genai.get_issue(first.issue_id).name == "Updated"
    assert [issue.name for issue in mlflow.genai.search_issues()] == ["Second"]


@pytest.mark.parametrize(
    "environment_variable", [None, "MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_NAME"]
)
def test_issue_api_uses_existing_experiment_resolution(monkeypatch, environment_variable):
    monkeypatch.setattr("mlflow.tracking.fluent._active_experiment_id", None)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    experiment_id = "0"
    if environment_variable is not None:
        experiment_id = mlflow.create_experiment("environment-issue-experiment")
        monkeypatch.setenv(
            environment_variable,
            experiment_id
            if environment_variable == "MLFLOW_EXPERIMENT_ID"
            else "environment-issue-experiment",
        )

    issue = mlflow.genai.create_issue(
        "Inferred", description="Inferred experiment", severity=IssueSeverity.HIGH
    )

    assert issue.experiment_id == experiment_id
    assert [result.issue_id for result in mlflow.genai.search_issues()] == [issue.issue_id]


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        (
            "create_issue",
            {
                "name": "Latency",
                "description": "Slow requests",
                "severity": IssueSeverity.HIGH,
                "experiment_id": "1",
            },
        ),
        ("get_issue", {"issue_id": ISSUE_ID, "experiment_id": "1"}),
        (
            "update_issue",
            {"issue_id": ISSUE_ID, "status": IssueStatus.RESOLVED, "experiment_id": "1"},
        ),
        ("search_issues", {"filter_string": "status = 'pending'", "experiment_id": "1"}),
    ],
)
def test_issue_api_preserves_client_errors(tracing_client, method, kwargs):
    error = MlflowException("Experiment access denied", "PERMISSION_DENIED")
    getattr(tracing_client, f"_{method}").side_effect = error

    with pytest.raises(MlflowException, match="Experiment access denied") as caught:
        getattr(mlflow.genai, method)(**kwargs)

    assert caught.value is error

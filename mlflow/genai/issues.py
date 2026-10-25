"""Fluent APIs for managing issues in MLflow experiments."""

from typing import Literal

from mlflow.entities.issue import Issue, IssueCategory, IssueSeverity, IssueStatus
from mlflow.store.entities.paged_list import PagedList
from mlflow.tracing.client import TracingClient
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.annotations import experimental


@experimental(version="3.18.0")
def create_issue(
    name: str,
    *,
    severity: IssueSeverity,
    description: str = "",
    status: IssueStatus = IssueStatus.PENDING,
    categories: list[IssueCategory | str] | None = None,
    last_seen_timestamp: int | None = None,
    source_run_id: str | None = None,
    metadata: dict[str, str] | None = None,
    issue_id: str | None = None,
    experiment_id: str | None = None,
) -> Issue:
    """Create an issue in an experiment.

    Args:
        name: Human-readable issue name.
        severity: Issue severity.
        description: Markdown description.
        status: Initial status.
        categories: Built-in categories or custom category strings.
        last_seen_timestamp: Most recent observation in milliseconds since the epoch.
        source_run_id: MLflow run that discovered the issue.
        metadata: Additional string metadata.
        issue_id: Optional caller-assigned issue ID.
        experiment_id: Experiment that owns the issue. Defaults to the active experiment.

    Returns:
        The created :py:class:`mlflow.entities.Issue`.
    """
    if experiment_id is None:
        experiment_id = _get_experiment_id()
    return TracingClient()._create_issue(
        experiment_id=experiment_id,
        name=name,
        description=description,
        status=status,
        severity=severity,
        categories=categories,
        last_seen_timestamp=last_seen_timestamp,
        source_run_id=source_run_id,
        metadata=metadata,
        issue_id=issue_id,
    )


@experimental(version="3.18.0")
def get_issue(issue_id: str) -> Issue:
    """Get an issue by ID.

    Args:
        issue_id: Issue ID.

    Returns:
        The :py:class:`mlflow.entities.Issue`.
    """
    return TracingClient()._get_issue(issue_id)


@experimental(version="3.18.0")
def update_issue(
    issue_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    status: IssueStatus | None = None,
    severity: IssueSeverity | None = None,
    categories: list[IssueCategory | str] | None = None,
    last_seen_timestamp: int | None = None,
    metadata: dict[str, str] | None = None,
    delete_metadata_keys: list[str] | None = None,
) -> Issue:
    """Update issue fields. Arguments set to ``None`` retain their values.

    Args:
        issue_id: Issue ID.
        name: New issue name.
        description: New Markdown description; ``""`` clears it.
        status: New status.
        severity: New severity.
        categories: Replacement categories; ``[]`` clears them.
        last_seen_timestamp: Most recent observation in milliseconds since the epoch.
        metadata: Metadata entries to add or replace.
        delete_metadata_keys: Metadata keys to remove.

    Returns:
        The updated :py:class:`mlflow.entities.Issue`.
    """
    return TracingClient()._update_issue(
        issue_id,
        name=name,
        description=description,
        status=status,
        severity=severity,
        categories=categories,
        last_seen_timestamp=last_seen_timestamp,
        metadata=metadata,
        delete_metadata_keys=delete_metadata_keys,
    )


@experimental(version="3.18.0")
def search_issues(
    filter_string: str | None = None,
    *,
    experiment_id: str | None = None,
    max_results: int | None = None,
    order_by: list[str] | None = None,
    page_token: str | None = None,
    view: Literal["basic", "full"] | None = None,
) -> PagedList[Issue]:
    """Search issues using MLflow filter syntax.

    Args:
        filter_string: SQL-like filter supported by the tracking store.
        experiment_id: Experiment to search. Defaults to the active experiment.
        max_results: Maximum results in one page.
        order_by: Ordering fields and optional ASC/DESC directions.
        page_token: Continuation token from a previous result.
        view: ``"basic"`` omits detailed content; ``"full"`` returns complete issues.

    Returns:
        A list of :py:class:`mlflow.entities.Issue` objects with a continuation ``token``
        and ``total_count`` when supplied by the store.
    """
    if experiment_id is None:
        experiment_id = _get_experiment_id()
    return TracingClient()._search_issues(
        experiment_id=experiment_id,
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
        view=view,
    )

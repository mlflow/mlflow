"""Fluent APIs for managing issues in MLflow experiments."""

from typing import Literal

from mlflow.entities.issue import Issue, IssueCategory, IssueSeverity, IssueStatus
from mlflow.entities.issue_detection_run import IssueDetectionRun
from mlflow.store.entities.paged_list import PagedList
from mlflow.tracing.client import TracingClient
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.annotations import experimental


@experimental(version="3.18.0")
def create_issue(
    name: str,
    *,
    severity: IssueSeverity,
    description: str,
    status: IssueStatus = IssueStatus.PENDING,
    categories: list[IssueCategory | str] | None = None,
    last_seen_timestamp: int | None = None,
    source_run_id: str | None = None,
    metadata: dict[str, str] | None = None,
    issue_id: str | None = None,
    experiment_id: str | None = None,
) -> Issue:
    """
    Create an issue in an experiment.

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
def get_issue(issue_id: str, *, experiment_id: str | None = None) -> Issue:
    """
    Get an issue by ID.

    Args:
        issue_id: Issue ID.
        experiment_id: Parent experiment for Databricks routing. Defaults to the active experiment.

    Returns:
        The :py:class:`mlflow.entities.Issue`.
    """
    if experiment_id is None:
        experiment_id = _get_experiment_id()
    return TracingClient()._get_issue(issue_id, experiment_id=experiment_id)


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
    experiment_id: str | None = None,
) -> Issue:
    """
    Update issue fields. Arguments set to ``None`` retain their values.

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
        experiment_id: Parent experiment for Databricks routing. Defaults to the active experiment.

    Returns:
        The updated :py:class:`mlflow.entities.Issue`.
    """
    if experiment_id is None:
        experiment_id = _get_experiment_id()
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
        experiment_id=experiment_id,
    )


@experimental(version="3.18.0")
def search_issues(
    filter_string: str | None = None,
    *,
    experiment_id: str | None = None,
    name: str | None = None,
    severities: list[IssueSeverity] | None = None,
    categories: list[IssueCategory | str] | None = None,
    max_results: int | None = None,
    order_by: list[str] | None = None,
    page_token: str | None = None,
    include_total_count: bool = False,
    view: Literal["basic", "full"] | None = None,
) -> PagedList[Issue]:
    """
    Search issues, combining different filter fields with AND.

    Args:
        filter_string: Comparisons on ``status`` or ``source_run_id`` using ``=`` or ``!=``,
            joined with AND, such as ``"status = 'pending'"``. Databricks supports only
            ``status`` filters.
        experiment_id: Experiment to search. Defaults to the active experiment.
        name: Name filter. SQL stores match a case-insensitive literal substring;
            Databricks uses its text search query.
        severities: Match any of these severities. Databricks supports low, medium, and high.
        categories: Match any of these built-in or custom categories. SQL stores ignore
            case and surrounding whitespace.
        max_results: Maximum results in one page. Defaults to 100 for SQL stores and 20
            for Databricks. Databricks accepts at most 100.
        order_by: Fields with optional ASC/DESC directions. SQL stores support ``name``,
            ``status``, ``severity``, ``created_timestamp``, ``last_updated_timestamp``, and
            ``last_seen_timestamp``. Databricks supports one field, limited to severity
            and timestamps. Defaults to the store's ordering.
        page_token: Continuation token from a previous result.
        include_total_count: Include the matching count before pagination in ``total_count``.
        view: ``"basic"`` returns an empty description. SQL stores also omit root causes
            and metadata; Databricks may omit large metadata entries. ``"full"`` returns
            complete issues. Defaults to ``"full"``.

    Returns:
        A list of :py:class:`mlflow.entities.Issue` objects with a continuation ``token``
        and ``total_count`` when supplied by the store.
    """
    if experiment_id is None:
        experiment_id = _get_experiment_id()
    return TracingClient()._search_issues(
        experiment_id=experiment_id,
        filter_string=filter_string,
        name=name,
        severities=severities,
        categories=categories,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
        include_total_count=include_total_count,
        view=view,
    )


@experimental(version="3.19.0")
def detect_issues(
    *,
    filter_string: str | None = None,
    trace_ids: list[str] | None = None,
    model: str | None = None,
    experiment_id: str | None = None,
) -> IssueDetectionRun:
    """
    Start an issue detection run over an experiment's traces.

    The run analyzes the traces in the background and files what it finds as issues, using the
    categories and instructions saved on the experiment. Poll it with
    :py:func:`get_issue_detection_run`. An experiment can have several detection runs in progress
    at once.

    On an MLflow tracking server, the run needs job execution enabled
    (``MLFLOW_SERVER_ENABLE_JOB_EXECUTION=true``). On Databricks, the experiment must store its
    traces in Unity Catalog, and the run reads them through the warehouse in
    ``MLFLOW_TRACING_SQL_WAREHOUSE_ID`` when it is set.

    .. code-block:: python

        import mlflow

        run = mlflow.genai.detect_issues(filter_string="trace.status = 'ERROR'")
        run = mlflow.genai.get_issue_detection_run(run.run_id)

    Args:
        filter_string: Restricts the run to traces matching this filter, in the
            :py:func:`mlflow.search_traces` filter syntax. When neither this nor ``trace_ids`` is
            set, the run looks at the experiment's newest traces. Either way it analyzes a sample,
            not every match: on an MLflow tracking server, up to 100 sessions picked at random
            from the newest 500 matching traces; on Databricks, the traces its agent's searches
            return.
        trace_ids: The IDs of the traces to analyze, all in the experiment and at most 500.
            Cannot be combined with ``filter_string``. Not supported on Databricks.
        model: The model to analyze traces with, as ``"<provider>:/<model>"`` or
            ``"gateway:/<endpoint>"``. Defaults to the model saved in the experiment's issue
            detection schedule. The run calls the model from the MLflow server, so a
            ``"<provider>:/<model>"`` model needs the provider's API key, such as
            ``OPENAI_API_KEY``, in the server's environment rather than the caller's. A
            ``"gateway:/<endpoint>"`` model uses the key saved with the AI Gateway endpoint. Not
            supported on Databricks, which chooses the model.
        experiment_id: The experiment whose traces to analyze. Defaults to the active experiment.

    Returns:
        The :py:class:`mlflow.entities.IssueDetectionRun`, which has just started.
    """
    if experiment_id is None:
        experiment_id = _get_experiment_id()
    return TracingClient()._detect_issues(
        experiment_id, filter_string=filter_string, model=model, trace_ids=trace_ids
    )


@experimental(version="3.19.0")
def get_issue_detection_run(run_id: str, *, experiment_id: str | None = None) -> IssueDetectionRun:
    """
    Get an issue detection run.

    Args:
        run_id: ID of the run, from :py:func:`detect_issues`.
        experiment_id: Parent experiment for Databricks routing. Defaults to the active
            experiment.

    Returns:
        The :py:class:`mlflow.entities.IssueDetectionRun`.
    """
    if experiment_id is None:
        experiment_id = _get_experiment_id()
    return TracingClient()._get_issue_detection_run(run_id, experiment_id=experiment_id)


@experimental(version="3.19.0")
def cancel_issue_detection_run(
    run_id: str, *, experiment_id: str | None = None
) -> IssueDetectionRun:
    """
    Cancel an issue detection run. Canceling a run that has ended returns it unchanged.

    Args:
        run_id: ID of the run, from :py:func:`detect_issues`.
        experiment_id: Parent experiment for Databricks routing. Defaults to the active
            experiment.

    Returns:
        The :py:class:`mlflow.entities.IssueDetectionRun`.
    """
    if experiment_id is None:
        experiment_id = _get_experiment_id()
    return TracingClient()._cancel_issue_detection_run(run_id, experiment_id=experiment_id)

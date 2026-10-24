"""Fluent APIs for managing experiment-scoped issues.

These APIs use the tracking URI configured by :py:func:`mlflow.set_tracking_uri`
and require a server exposing the experiment-scoped Issues REST APIs.
"""

from typing import Any
from urllib.parse import quote

from mlflow.entities.issue_resource import IssueResource
from mlflow.exceptions import MlflowException, MlflowNotImplementedException
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import NOT_SET
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracking import get_tracking_uri
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.utils.annotations import experimental
from mlflow.utils.rest_utils import http_request, verify_rest_response


def _metadata_mask_path(key: str) -> str:
    if not isinstance(key, str) or not key or "," in key or "`" in key:
        raise MlflowException.invalid_parameter_value(
            "Metadata keys in an update mask cannot be empty or contain commas or backticks."
        )
    # Quoting every key also handles dots without treating them as nested fields.
    return f"metadata.`{key}`"


def _path(experiment_id: str, issue_id: str | None = None) -> str:
    if not isinstance(experiment_id, str) or not experiment_id:
        raise MlflowException.invalid_parameter_value("experiment_id must be a nonempty string")
    path = f"/api/2.0/mlflow/experiments/{quote(experiment_id, safe='')}/issues"
    if issue_id is not None:
        if not isinstance(issue_id, str) or not issue_id:
            raise MlflowException.invalid_parameter_value("issue_id must be a nonempty string")
        path += f"/{quote(issue_id, safe='')}"
    return path


def _request(method: str, path: str, *, body=None, params=None) -> dict[str, Any]:
    store = _get_store(get_tracking_uri())
    if not isinstance(store, RestStore):
        raise MlflowNotImplementedException(
            "Issue APIs require a tracking server that exposes the experiment-scoped "
            "Issues REST APIs. Local tracking stores are not supported."
        )
    response = http_request(
        host_creds=store.get_host_creds(),
        endpoint=path,
        method=method,
        params=params,
        json=body,
    )
    expected_status = 204 if method == "DELETE" and response.status_code == 204 else 200
    response = verify_rest_response(response, path, expected_status=expected_status)
    return response.json() if response.content else {}


@experimental(version="3.18.0")
def create_issue(
    experiment_id: str,
    title: str,
    *,
    severity: str,
    description: str = "",
    status: str = "PENDING",
    categories: list[str] | None = None,
    tags: dict[str, str] | None = None,
    last_seen_time: str | None = None,
    source_type: str = "MANUAL",
    source_id: str | None = None,
    metadata: dict[str, str] | None = None,
    issue_id: str | None = None,
) -> IssueResource:
    """Create an issue in an experiment.

    Args:
        experiment_id: Experiment that owns the issue.
        title: Human-readable issue title.
        severity: ``LOW``, ``MEDIUM``, or ``HIGH``.
        description: Markdown description of the issue.
        status: ``PENDING``, ``RESOLVED``, or ``REJECTED``.
        categories: Issue category labels.
        tags: String tags that can be used to filter issues.
        last_seen_time: Most recent observation as an RFC 3339 timestamp.
        source_type: ``MANUAL``, ``ADHOC_RUN``, ``SCHEDULED_RUN``, or
            ``SINGLE_TRACE_ANALYSIS``. Source fields cannot be changed after creation.
        source_id: Identifier of the source that produced the issue.
        metadata: Additional string metadata.
        issue_id: Optional caller-assigned ID, ``iss-`` followed by 32 lowercase hex
            characters. Reusing an ID raises a conflict.

    Returns:
        The created :py:class:`mlflow.entities.IssueResource`.

    Example:
        .. code-block:: python

            import mlflow

            issue = mlflow.genai.create_issue(
                experiment_id="1",
                title="Payment retries fail",
                severity="HIGH",
                categories=["correctness"],
            )
            print(issue.issue_id, issue.title)
    """
    path = _path(experiment_id)
    body = {
        key: value
        for key, value in {
            "title": title,
            "severity": severity,
            "description": description,
            "status": status,
            "categories": categories,
            "tags": tags,
            "last_seen_time": last_seen_time,
            "source_type": source_type,
            "source_id": source_id,
            "metadata": metadata,
        }.items()
        if value is not None
    }
    resource = _request(
        "POST", path, body=body, params={"issue_id": issue_id} if issue_id is not None else None
    )
    return IssueResource.from_dictionary(resource)


@experimental(version="3.18.0")
def get_issue(experiment_id: str, issue_id: str) -> IssueResource:
    """Retrieve the complete issue from its owning experiment.

    Args:
        experiment_id: Experiment that owns the issue.
        issue_id: Issue ID to retrieve.

    Returns:
        The :py:class:`mlflow.entities.IssueResource`, including its description and metadata.
    """
    path = _path(experiment_id, issue_id)
    resource = _request("GET", path)
    return IssueResource.from_dictionary(resource)


@experimental(version="3.18.0")
def update_issue(
    experiment_id: str,
    issue_id: str,
    *,
    title: str = NOT_SET,
    description: str | None = NOT_SET,
    status: str = NOT_SET,
    severity: str = NOT_SET,
    categories: list[str] | None = NOT_SET,
    tags: dict[str, str] | None = NOT_SET,
    last_seen_time: str | None = NOT_SET,
    metadata: dict[str, str] | None = None,
    delete_metadata_keys: list[str] | None = None,
) -> IssueResource:
    """Update only the issue fields explicitly supplied.

    The SDK builds the REST update mask. Omitted fields retain their values;
    ``None`` clears an optional field. Metadata is updated per key, so unrelated
    entries are preserved.

    Args:
        experiment_id: Experiment that owns the issue.
        issue_id: Issue ID to update.
        title: New human-readable title.
        description: New Markdown description, or ``None`` to clear it.
        status: ``PENDING``, ``RESOLVED``, or ``REJECTED``.
        severity: ``LOW``, ``MEDIUM``, or ``HIGH``.
        categories: Replacement category labels; ``[]`` or ``None`` clears them.
        tags: Replacement tags; ``{}`` or ``None`` clears them.
        last_seen_time: RFC 3339 timestamp. ``None`` resets a manual issue to its
            creation time; issues produced by detection must retain an observation time.
        metadata: Metadata entries to add or replace.
        delete_metadata_keys: Metadata keys to remove. A key cannot be updated and
            deleted in the same request.

    Returns:
        The updated :py:class:`mlflow.entities.IssueResource`.
    """
    path = _path(experiment_id, issue_id)
    fields = {
        "title": title,
        "description": description,
        "status": status,
        "severity": severity,
        "categories": categories,
        "tags": tags,
        "last_seen_time": last_seen_time,
    }
    paths = [key for key, value in fields.items() if value is not NOT_SET]
    body = {
        key: value for key, value in fields.items() if value is not NOT_SET and value is not None
    }
    if metadata is not None and not isinstance(metadata, dict):
        raise MlflowException.invalid_parameter_value("metadata must be a dictionary")
    if delete_metadata_keys is not None and (
        not isinstance(delete_metadata_keys, list)
        or any(not isinstance(key, str) for key in delete_metadata_keys)
    ):
        raise MlflowException.invalid_parameter_value(
            "delete_metadata_keys must be a list of strings"
        )
    if set(metadata or {}) & set(delete_metadata_keys or []):
        raise MlflowException.invalid_parameter_value(
            "Cannot update and delete the same metadata key"
        )
    paths.extend(
        _metadata_mask_path(key)
        for key in dict.fromkeys([*(metadata or {}), *(delete_metadata_keys or [])])
    )
    if metadata:
        body["metadata"] = metadata
    body["name"] = f"experiments/{experiment_id}/issues/{issue_id}"
    update_mask = ",".join(paths)
    if not paths:
        raise MlflowException.invalid_parameter_value("At least one issue field must be updated")
    resource = _request("PATCH", path, body=body, params={"update_mask": update_mask})
    return IssueResource.from_dictionary(resource)


@experimental(version="3.18.0")
def search_issues(
    experiment_id: str,
    *,
    query: str | None = None,
    statuses: list[str] | None = None,
    severities: list[str] | None = None,
    categories: list[str] | None = None,
    tag_filters: list[dict[str, Any]] | None = None,
    sort_key: str | None = None,
    sort_order: str | None = None,
    page_size: int | None = None,
    page_token: str | None = None,
    view: str | None = None,
) -> PagedList[IssueResource]:
    """Search one experiment's issues with filters and pagination.

    Values within each filter are ORed; different filters are ANDed.

    Args:
        experiment_id: Experiment to search.
        query: Case-insensitive substring of the issue title.
        statuses: ``PENDING``, ``RESOLVED``, or ``REJECTED`` values to match.
        severities: ``LOW``, ``MEDIUM``, or ``HIGH`` values to match.
        categories: Category labels to match.
        tag_filters: Filters such as ``[{"key": "team", "values": ["payments"]}]``.
            Omitting ``values`` matches any issue with the specified key.
        sort_key: ``SEVERITY``, ``LAST_SEEN_TIME``, ``UPDATE_TIME``, or ``CREATE_TIME``.
        sort_order: ``DESCENDING`` or ``ASCENDING``. Issues are grouped by status first.
        page_size: Page size; defaults to 20, with a maximum of 100.
        page_token: Token from a previous result's ``token`` attribute.
        view: ``BASIC`` (default) omits the description and trace-list metadata;
            ``FULL`` returns complete issues.

    Returns:
        A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
        :py:class:`mlflow.entities.IssueResource` objects. ``token`` identifies the next
        page, and ``total_count`` is the number of matching issues across all pages.
    """
    path = _path(experiment_id) + ":search"
    body = {
        key: value
        for key, value in {
            "parent": f"experiments/{experiment_id}",
            "query": query,
            "statuses": statuses,
            "severities": severities,
            "categories": categories,
            "tag_filters": tag_filters,
            "sort_key": sort_key,
            "sort_order": sort_order,
            "page_size": page_size,
            "page_token": page_token,
            "view": view,
        }.items()
        if value is not None
    }
    response = _request("POST", path, body=body)
    return PagedList(
        [IssueResource.from_dictionary(issue) for issue in response.get("issues", [])],
        token=response.get("next_page_token") or None,
        total_count=response.get("total_size"),
    )


@experimental(version="3.18.0")
def delete_issue(
    experiment_id: str, issue_id: str, *, force: bool = False, sql_warehouse_id: str | None = None
) -> None:
    """Delete an issue from its owning experiment.

    On Databricks, deletion fails while assessments on Unity Catalog traces reference
    the issue unless ``force=True``. Forced deletion removes those references first
    and requires a SQL warehouse.

    Args:
        experiment_id: Experiment that owns the issue.
        issue_id: Issue ID to delete.
        force: Remove Unity Catalog trace references before deleting the issue.
        sql_warehouse_id: Databricks SQL warehouse used for Unity Catalog trace cleanup.
    """
    path = _path(experiment_id, issue_id)
    if type(force) is not bool:
        raise MlflowException.invalid_parameter_value("force must be a boolean")
    params = {"force": str(force).lower()}
    if sql_warehouse_id is not None:
        params["sql_warehouse_id"] = sql_warehouse_id
    _request("DELETE", path, params=params)

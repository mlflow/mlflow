from typing import Any
from urllib.parse import quote

from mlflow.entities.issue_resource import IssueResource
from mlflow.exceptions import MlflowException, MlflowNotImplementedException
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import NOT_SET
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracking._tracking_service.utils import _get_store, _resolve_tracking_uri
from mlflow.utils.rest_utils import http_request, verify_rest_response


def _metadata_mask_path(key: str) -> str:
    if not isinstance(key, str) or not key or "," in key or "`" in key:
        raise MlflowException.invalid_parameter_value(
            "Metadata keys in an update mask cannot be empty or contain commas or backticks."
        )
    # Quoting every key also handles dots without treating them as nested fields.
    return f"metadata.`{key}`"


class IssueClient:
    """Client for experiment-scoped Issues REST APIs."""

    def __init__(self, tracking_uri: str | None = None):
        self.tracking_uri = _resolve_tracking_uri(tracking_uri)

    @property
    def store(self):
        store = _get_store(self.tracking_uri)
        if not isinstance(store, RestStore):
            raise MlflowNotImplementedException(
                "Issue SDK methods require a tracking server that exposes the experiment-scoped "
                "Issues REST APIs. Local tracking stores are not supported."
            )
        return store

    @staticmethod
    def _path(experiment_id: str, issue_id: str | None = None) -> str:
        if not isinstance(experiment_id, str) or not experiment_id:
            raise MlflowException.invalid_parameter_value("experiment_id must be a nonempty string")
        path = f"/api/2.0/mlflow/experiments/{quote(experiment_id, safe='')}/issues"
        if issue_id is not None:
            if not isinstance(issue_id, str) or not issue_id:
                raise MlflowException.invalid_parameter_value("issue_id must be a nonempty string")
            path += f"/{quote(issue_id, safe='')}"
        return path

    def _request(self, method: str, path: str, *, body=None, params=None) -> dict[str, Any]:
        response = http_request(
            host_creds=self.store.get_host_creds(),
            endpoint=path,
            method=method,
            params=params,
            json=body,
        )
        expected_status = 204 if method == "DELETE" and response.status_code == 204 else 200
        response = verify_rest_response(response, path, expected_status=expected_status)
        return response.json() if response.content else {}

    def create_issue(
        self,
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
        path = self._path(experiment_id)
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
        resource = self._request(
            "POST",
            path,
            body=body,
            params={"issue_id": issue_id} if issue_id is not None else None,
        )
        return IssueResource.from_dictionary(resource)

    def get_issue(self, experiment_id: str, issue_id: str) -> IssueResource:
        path = self._path(experiment_id, issue_id)
        resource = self._request("GET", path)
        return IssueResource.from_dictionary(resource)

    def update_issue(
        self,
        experiment_id: str,
        issue_id: str,
        *,
        title=NOT_SET,
        description=NOT_SET,
        status=NOT_SET,
        severity=NOT_SET,
        categories=NOT_SET,
        tags=NOT_SET,
        last_seen_time=NOT_SET,
        metadata: dict[str, str] | None = None,
        delete_metadata_keys: list[str] | None = None,
    ) -> IssueResource:
        path = self._path(experiment_id, issue_id)
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
            key: value
            for key, value in fields.items()
            if value is not NOT_SET and value is not None
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
            raise MlflowException.invalid_parameter_value(
                "At least one issue field must be updated"
            )
        resource = self._request("PATCH", path, body=body, params={"update_mask": update_mask})
        return IssueResource.from_dictionary(resource)

    def delete_issue(
        self,
        experiment_id: str,
        issue_id: str,
        *,
        force: bool = False,
        sql_warehouse_id: str | None = None,
    ) -> None:
        path = self._path(experiment_id, issue_id)
        if type(force) is not bool:
            raise MlflowException.invalid_parameter_value("force must be a boolean")
        params = {"force": str(force).lower()}
        if sql_warehouse_id is not None:
            params["sql_warehouse_id"] = sql_warehouse_id
        self._request("DELETE", path, params=params)

    def search_issues(
        self,
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
        path = self._path(experiment_id) + ":search"
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
        response = self._request("POST", path, body=body)
        return PagedList(
            [IssueResource.from_dictionary(issue) for issue in response.get("issues", [])],
            token=response.get("next_page_token") or None,
            total_count=response.get("total_size"),
        )

from __future__ import annotations

from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.utils.rest_utils import http_request
from mlflow.utils.uri import is_databricks_uri
from mlflow.utils.workspace_context import get_request_workspace


class WorkspaceRestStoreMixin:
    """
    Shared workspace capability detection for REST-based stores.
    """

    _SERVER_INFO_ENDPOINT = "/api/3.0/mlflow/server-info"
    _WORKSPACE_UNSUPPORTED_ERROR = (
        "Active workspace '{workspace}' cannot be used because the remote server does not "
        "support workspaces. Restart the server with --enable-workspaces or unset the active "
        "workspace."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._workspace_support: bool | None = None

    @property
    def supports_workspaces(self) -> bool:
        if self._workspace_support is not None:
            return self._workspace_support

        host_creds = self.get_host_creds()
        store_uri = getattr(host_creds, "host", None)
        if store_uri and is_databricks_uri(store_uri):
            self._workspace_support = False
            return False

        supported = self._probe_workspace_support()
        self._workspace_support = supported
        return supported

    def _validate_workspace_support_if_specified(self) -> None:
        """
        Raise an error if a workspace is active but the server doesn't support workspaces.
        """
        workspace = get_request_workspace()
        if workspace is None:
            return
        if not self.supports_workspaces:
            raise MlflowException(
                self._WORKSPACE_UNSUPPORTED_ERROR.format(workspace=workspace),
                error_code=databricks_pb2.FEATURE_DISABLED,
            )

    def _probe_workspace_support(self) -> bool:
        host_creds = self.get_host_creds()
        try:
            response = http_request(
                host_creds=host_creds,
                endpoint=self._SERVER_INFO_ENDPOINT,
                method="GET",
                timeout=3,
                max_retries=0,
                raise_on_status=False,
            )
        except Exception as exc:  # pragma: no cover - network errors vary
            raise MlflowException(
                message=f"Failed to query {self._SERVER_INFO_ENDPOINT}: {exc}",
                error_code=databricks_pb2.INTERNAL_ERROR,
            ) from exc

        if response.status_code == 404:
            # This is expected for older servers that don't have the server-info endpoint.
            return False

        if response.status_code != 200:
            raise MlflowException(
                message=(
                    f"Failed to query {self._SERVER_INFO_ENDPOINT}: "
                    f"{response.status_code} {response.text}"
                ),
                error_code=databricks_pb2.TEMPORARILY_UNAVAILABLE,
            )

        return response.json().get("workspaces_enabled", False)

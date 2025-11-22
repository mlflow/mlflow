import logging
from abc import ABCMeta, abstractmethod

from mlflow.environment_variables import MLFLOW_REGISTRY_URI
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.utils import rest_utils
from mlflow.utils.rest_utils import call_endpoint, call_endpoints, http_request
from mlflow.utils.uri import is_databricks_uri

_logger = logging.getLogger(__name__)


class BaseRestStore(AbstractStore):
    """
    Base class client for a remote model registry server accessed via REST API calls
    """

    __metaclass__ = ABCMeta
    _SERVER_FEATURES_ENDPOINT = "/api/2.0/mlflow/server-features"

    def __init__(self, get_host_creds):
        super().__init__()
        self.get_host_creds = get_host_creds
        self._workspace_support: bool | None = None

    def supports_workspaces(self) -> bool:
        if self._workspace_support is not None:
            return self._workspace_support

        registry_uri = MLFLOW_REGISTRY_URI.get()
        if registry_uri and is_databricks_uri(registry_uri):
            self._workspace_support = False
            return False

        supported = self._probe_workspace_support()
        self._workspace_support = supported
        return supported

    def _probe_workspace_support(self) -> bool:
        host_creds = self.get_host_creds()
        try:
            response = http_request(
                host_creds=host_creds,
                endpoint=self._SERVER_FEATURES_ENDPOINT,
                method="GET",
                timeout=3,
                max_retries=0,
                raise_on_status=False,
            )
        except Exception as exc:  # pragma: no cover - network errors vary
            raise MlflowException(
                message=f"Failed to query {self._SERVER_FEATURES_ENDPOINT}: {exc}",
                error_code=databricks_pb2.INTERNAL_ERROR,
            ) from exc

        if response.status_code == 404:
            return False

        if response.status_code >= 400:
            raise MlflowException(
                message=(
                    f"Failed to query {self._SERVER_FEATURES_ENDPOINT}: "
                    f"{response.status_code} {response.text}"
                ),
                error_code=databricks_pb2.UNAVAILABLE,
            )

        return response.json().get("workspaces_enabled", False)

    @abstractmethod
    def _get_all_endpoints_from_method(self, method):
        pass

    @abstractmethod
    def _get_endpoint_from_method(self, method):
        pass

    @abstractmethod
    def _get_response_from_method(self, method):
        pass

    def _call_endpoint(self, api, json_body, call_all_endpoints=False, extra_headers=None):
        response_proto = self._get_response_from_method(api)
        workspace = rest_utils._resolve_active_workspace()
        if isinstance(workspace, str):
            workspace = workspace.strip()
            if not workspace:
                workspace = None
        workspace_requested = workspace is not None
        if workspace_requested and not self.supports_workspaces():
            raise MlflowException(
                f"Active workspace '{workspace}' cannot be used because the remote model registry "
                "server does not support workspaces. Restart the server with --enable-workspaces "
                "or unset the active workspace.",
                error_code=databricks_pb2.FEATURE_DISABLED,
            )
        if call_all_endpoints:
            endpoints = self._get_all_endpoints_from_method(api)
            return call_endpoints(
                self.get_host_creds(), endpoints, json_body, response_proto, extra_headers
            )
        else:
            endpoint, method = self._get_endpoint_from_method(api)
            return call_endpoint(
                self.get_host_creds(), endpoint, method, json_body, response_proto, extra_headers
            )

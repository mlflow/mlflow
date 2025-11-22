import logging
from abc import ABCMeta, abstractmethod

from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.utils import rest_utils
from mlflow.utils.rest_utils import call_endpoint, call_endpoints

_logger = logging.getLogger(__name__)


class BaseRestStore(AbstractStore):
    """
    Base class client for a remote model registry server accessed via REST API calls
    """

    __metaclass__ = ABCMeta

    def __init__(self, get_host_creds):
        super().__init__()
        self.get_host_creds = get_host_creds
        self._workspace_support: bool | None = None

    def supports_workspaces(self) -> bool:
        if self._workspace_support is None:
            self._workspace_support = self._probe_workspace_support()
        return self._workspace_support

    def _probe_workspace_support(self) -> bool:
        host_creds = self.get_host_creds()
        try:
            response = rest_utils.http_request(
                host_creds=host_creds,
                endpoint="/api/2.0/mlflow/workspaces",
                method="GET",
                timeout=3,
                max_retries=0,
                raise_on_status=False,
            )
        except Exception as exc:  # pragma: no cover - network errors vary
            _logger.debug("Failed to probe workspace support: %s", exc)
            return False

        if response.status_code in (200, 401, 403):
            return True

        _logger.debug(
            "Workspace endpoint probe returned status %s: %s",
            response.status_code,
            response.text,
        )
        return False

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

from __future__ import annotations

import threading
import warnings
from functools import lru_cache, partial

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.tracking.registry import StoreRegistry
from mlflow.utils.credentials import get_default_host_creds
from mlflow.utils.plugins import get_entry_points

_WORKSPACE_PROVIDER_ENTRYPOINT_GROUP = "mlflow.workspace_provider"
_building_workspace_store_lock = threading.RLock()


class UnsupportedWorkspaceStoreURIException(MlflowException):
    """Exception thrown when building a workspace store with an unsupported URI."""

    def __init__(self, unsupported_uri, supported_uri_schemes):
        message = (
            "Workspace functionality is unavailable; got unsupported URI "
            f"'{unsupported_uri}' for the workspace backend. Supported URI schemes are: "
            f"{supported_uri_schemes}. See the workspace configuration guide for instructions."
        )
        super().__init__(message, error_code=INVALID_PARAMETER_VALUE)
        self.supported_uri_schemes = supported_uri_schemes


class WorkspaceStoreRegistry(StoreRegistry):
    """Scheme-based registry for workspace store implementations."""

    def __init__(self):
        super().__init__(_WORKSPACE_PROVIDER_ENTRYPOINT_GROUP)

    def register_entrypoints(self):
        """Register workspace stores provided by other packages."""
        for entrypoint in get_entry_points(self.group_name):
            try:
                self.register(entrypoint.name, entrypoint.load())
            except (AttributeError, ImportError) as exc:
                warnings.warn(
                    f"Failure attempting to register workspace provider '{entrypoint.name}': {exc}",
                    stacklevel=2,
                )

    def get_store(self, workspace_uri: str):
        """Return a workspace store instance for the provided URI."""

        return self._get_store_with_resolved_uri(workspace_uri)

    @lru_cache(maxsize=100)
    def _get_store_with_resolved_uri(self, workspace_uri: str):
        with _building_workspace_store_lock:
            try:
                builder = self.get_store_builder(workspace_uri)
            except MlflowException as exc:
                raise UnsupportedWorkspaceStoreURIException(
                    unsupported_uri=workspace_uri,
                    supported_uri_schemes=list(self._registry.keys()),
                ) from exc
            return builder(workspace_uri=workspace_uri)


_workspace_store_registry: WorkspaceStoreRegistry | None = None


def _get_workspace_store_registry() -> WorkspaceStoreRegistry:
    global _workspace_store_registry
    if _workspace_store_registry is None:
        with _building_workspace_store_lock:
            # Double-check to avoid redundant initialization when multiple threads race.
            if _workspace_store_registry is None:
                registry = WorkspaceStoreRegistry()
                _register_default_workspace_stores(registry)
                registry.register_entrypoints()
                _workspace_store_registry = registry
    return _workspace_store_registry


def _get_sqlalchemy_workspace_store(workspace_uri: str):
    from mlflow.store.workspace.sqlalchemy_store import SqlAlchemyStore

    return SqlAlchemyStore(workspace_uri)


def _get_rest_workspace_store(workspace_uri: str):
    from mlflow.store.workspace.rest_store import RestWorkspaceStore

    return RestWorkspaceStore(partial(get_default_host_creds, workspace_uri))


def _register_default_workspace_stores(registry: WorkspaceStoreRegistry) -> None:
    # Register SQLAlchemy builder for common database schemes.
    for scheme in DATABASE_ENGINES:
        registry.register(scheme, _get_sqlalchemy_workspace_store)

    # Register REST-based workspace stores.
    for scheme in ["http", "https"]:
        registry.register(scheme, _get_rest_workspace_store)


def get_workspace_store(workspace_uri: str):
    """
    Return a workspace store for the specified workspace URI.

    Args:
        workspace_uri: Workspace backend URI.
    """

    return _get_workspace_store_registry().get_store(workspace_uri=workspace_uri)

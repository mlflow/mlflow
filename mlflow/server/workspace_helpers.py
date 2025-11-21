from __future__ import annotations

import logging
import os

from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES, MLFLOW_WORKSPACE_URI
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.tracking._workspace import utils as workspace_utils
from mlflow.tracking._workspace.registry import get_workspace_store

_logger = logging.getLogger(__name__)

_workspace_store = None


def _get_workspace_store(workspace_uri: str | None = None, tracking_uri: str | None = None):
    """
    Resolve and cache the workspace store configured for this server process.

    The store is constructed on first invocation using the provided arguments (or their
    environment-derived defaults) and memoized for all subsequent calls, regardless of any new
    ``workspace_uri`` / ``tracking_uri`` values supplied later.
    """
    if not MLFLOW_ENABLE_WORKSPACES.get():
        raise MlflowException(
            "Workspace APIs are not available: multi-tenancy is not enabled on this server",
            databricks_pb2.FEATURE_DISABLED,
        )

    global _workspace_store
    if _workspace_store is not None:
        return _workspace_store

    from mlflow.server import BACKEND_STORE_URI_ENV_VAR

    resolved_tracking_uri = tracking_uri or os.environ.get(BACKEND_STORE_URI_ENV_VAR)
    resolved_workspace_uri = workspace_utils.resolve_workspace_uri(
        workspace_uri, tracking_uri=resolved_tracking_uri
    )
    if resolved_workspace_uri is None:
        raise MlflowException.invalid_parameter_value(
            "Workspace URI could not be resolved. Provide --workspace-store-uri or set "
            f"{MLFLOW_WORKSPACE_URI.name}."
        )

    _workspace_store = get_workspace_store(workspace_uri=resolved_workspace_uri)
    return _workspace_store


__all__ = ["_get_workspace_store"]

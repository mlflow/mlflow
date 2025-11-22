from __future__ import annotations

import threading
from typing import Callable, TypeVar

from mlflow.entities.workspace import Workspace
from mlflow.environment_variables import MLFLOW_WORKSPACE
from mlflow.exceptions import MlflowException, RestException
from mlflow.protos import databricks_pb2
from mlflow.protos.databricks_pb2 import FEATURE_DISABLED
from mlflow.store.workspace.abstract_store import WorkspaceNameValidator
from mlflow.tracking._workspace.context import clear_workspace, set_current_workspace
from mlflow.tracking.client import MlflowClient
from mlflow.utils.annotations import experimental
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

T = TypeVar("T")

_workspace_lock = threading.Lock()


def _workspace_client_call(func: Callable[[MlflowClient], T]) -> T:
    client = MlflowClient()
    try:
        return func(client)
    except RestException as exc:
        if exc.error_code == databricks_pb2.ErrorCode.Name(databricks_pb2.ENDPOINT_NOT_FOUND):
            raise MlflowException(
                "The configured tracking server does not expose workspace APIs. "
                "Ensure multi-tenancy is enabled.",
                error_code=FEATURE_DISABLED,
            ) from exc
        raise


@experimental(version="3.7.0")
def set_workspace(workspace: str | None) -> None:
    """Set the active workspace for subsequent MLflow operations."""

    with _workspace_lock:
        if workspace is None:
            clear_workspace()
            MLFLOW_WORKSPACE.unset()
            return

        if workspace != DEFAULT_WORKSPACE_NAME:
            WorkspaceNameValidator.validate(workspace)

        set_current_workspace(workspace)
        MLFLOW_WORKSPACE.set(workspace)


@experimental(version="3.7.0")
def list_workspaces() -> list[Workspace]:
    """Return the list of workspaces available to the current user."""

    return _workspace_client_call(lambda client: client.list_workspaces())


@experimental(version="3.7.0")
def get_workspace(name: str) -> Workspace:
    """Return metadata for the specified workspace."""

    return _workspace_client_call(lambda client: client.get_workspace(name))


@experimental(version="3.7.0")
def create_workspace(name: str, description: str | None = None) -> Workspace:
    """Create a new workspace."""

    WorkspaceNameValidator.validate(name)
    return _workspace_client_call(
        lambda client: client.create_workspace(name=name, description=description)
    )


@experimental(version="3.7.0")
def update_workspace(name: str, description: str | None = None) -> Workspace:
    """Update metadata for an existing workspace."""

    if name != DEFAULT_WORKSPACE_NAME:
        WorkspaceNameValidator.validate(name)
    return _workspace_client_call(
        lambda client: client.update_workspace(name=name, description=description)
    )


@experimental(version="3.7.0")
def delete_workspace(name: str) -> None:
    """Delete an existing workspace."""

    if name != DEFAULT_WORKSPACE_NAME:
        WorkspaceNameValidator.validate(name)
    _workspace_client_call(lambda client: client.delete_workspace(name=name))


__all__ = [
    "Workspace",
    "set_workspace",
    "list_workspaces",
    "get_workspace",
    "create_workspace",
    "update_workspace",
    "delete_workspace",
]

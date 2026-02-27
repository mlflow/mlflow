from __future__ import annotations

import threading
from typing import Callable, TypeVar

from mlflow.entities.workspace import Workspace, WorkspaceDeletionMode
from mlflow.exceptions import MlflowException, RestException
from mlflow.protos import databricks_pb2
from mlflow.protos.databricks_pb2 import FEATURE_DISABLED
from mlflow.store.workspace.abstract_store import WorkspaceNameValidator
from mlflow.tracking.client import MlflowClient
from mlflow.utils.annotations import experimental
from mlflow.utils.workspace_context import set_workspace as set_context_workspace
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
                "Ensure workspace is enabled.",
                error_code=FEATURE_DISABLED,
            ) from exc
        raise


@experimental(version="3.10.0")
def set_workspace(workspace: str | None) -> None:
    """Set the active workspace for subsequent MLflow operations."""

    with _workspace_lock:
        if workspace is None:
            set_context_workspace(None)
            return

        if workspace != DEFAULT_WORKSPACE_NAME:
            WorkspaceNameValidator.validate(workspace)

        set_context_workspace(workspace)


@experimental(version="3.10.0")
def list_workspaces() -> list[Workspace]:
    """Return the list of workspaces available to the current user."""

    return _workspace_client_call(lambda client: client.list_workspaces())


@experimental(version="3.10.0")
def get_workspace(name: str) -> Workspace:
    """Return metadata for the specified workspace."""

    return _workspace_client_call(lambda client: client.get_workspace(name))


@experimental(version="3.10.0")
def create_workspace(
    name: str, description: str | None = None, default_artifact_root: str | None = None
) -> Workspace:
    """Create a new workspace.

    Args:
        name: The workspace name (lowercase alphanumeric with optional internal hyphens).
        description: Optional description of the workspace.
        default_artifact_root: Optional artifact root URI; falls back to server default.

    Returns:
        The newly created workspace.

    Raises:
        MlflowException: If the name is invalid, already exists, or no artifact root available.
    """
    WorkspaceNameValidator.validate(name)
    return _workspace_client_call(
        lambda client: client.create_workspace(
            name=name,
            description=description,
            default_artifact_root=default_artifact_root,
        )
    )


@experimental(version="3.10.0")
def update_workspace(
    name: str, description: str | None = None, default_artifact_root: str | None = None
) -> Workspace:
    """Update metadata for an existing workspace.

    Args:
        name: The name of the workspace to update.
        description: New description, or ``None`` to leave unchanged.
        default_artifact_root: New artifact root URI, empty string to clear, or ``None``.

    Returns:
        The updated workspace.

    Raises:
        MlflowException: If the workspace does not exist or no artifact root available.
    """
    if name != DEFAULT_WORKSPACE_NAME:
        WorkspaceNameValidator.validate(name)
    return _workspace_client_call(
        lambda client: client.update_workspace(
            name=name,
            description=description,
            default_artifact_root=default_artifact_root,
        )
    )


@experimental(version="3.10.0")
def delete_workspace(name: str, *, mode: str = WorkspaceDeletionMode.RESTRICT) -> None:
    """Delete an existing workspace.

    Args:
        name: Name of the workspace to delete.
        mode: Deletion mode. One of SET_DEFAULT, CASCADE, or RESTRICT.
    """
    try:
        deletion_mode = WorkspaceDeletionMode(mode)
    except ValueError:
        raise MlflowException.invalid_parameter_value(
            f"Invalid deletion mode '{mode}'. "
            f"Must be one of: {', '.join(m.value for m in WorkspaceDeletionMode)}"
        )
    if name != DEFAULT_WORKSPACE_NAME:
        WorkspaceNameValidator.validate(name)
    _workspace_client_call(lambda client: client.delete_workspace(name=name, mode=deletion_mode))


__all__ = [
    "Workspace",
    "set_workspace",
    "list_workspaces",
    "get_workspace",
    "create_workspace",
    "update_workspace",
    "delete_workspace",
]

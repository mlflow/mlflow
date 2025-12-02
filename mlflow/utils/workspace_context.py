from __future__ import annotations

from contextvars import ContextVar, Token

from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

_WORKSPACE: ContextVar[str | None] = ContextVar("mlflow_active_workspace", default=None)


def get_request_workspace() -> str | None:
    """
    Return the workspace currently bound to this execution context.

    This helper is request-scoped and should be used by code paths that need to
    know which workspace initiated the current request.
    """

    return _WORKSPACE.get()


def set_current_workspace(workspace: str | None) -> Token[str | None]:
    """Bind the given workspace name to the current execution context."""

    if workspace is not None and workspace != DEFAULT_WORKSPACE_NAME:
        from mlflow.store.workspace.abstract_store import WorkspaceNameValidator

        WorkspaceNameValidator.validate(workspace)

    return _WORKSPACE.set(workspace)


def reset_workspace(token: Token[str | None]) -> None:
    """Restore the workspace context to the state captured by ``token``."""
    _WORKSPACE.reset(token)


def clear_workspace() -> None:
    """Explicitly clear the current workspace binding (set it to ``None``)."""
    _WORKSPACE.set(None)


class WorkspaceContext:
    """Context manager helper that temporarily sets the active workspace."""

    def __init__(self, workspace: str | None):
        self._workspace = workspace
        self._token: Token[str | None] | None = None

    def __enter__(self) -> str | None:
        self._token = set_current_workspace(self._workspace)
        return self._workspace

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            reset_workspace(self._token)
            self._token = None

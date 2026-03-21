from __future__ import annotations

from contextvars import ContextVar, Token

from mlflow.environment_variables import MLFLOW_WORKSPACE
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

_WORKSPACE: ContextVar[str | None] = ContextVar("mlflow_active_workspace", default=None)
_IS_WORKSPACE_RESOLVED: ContextVar[bool] = ContextVar("mlflow_workspace_resolved", default=False)


def get_request_workspace() -> str | None:
    """
    Return the active workspace for the current execution context.

    Resolution order:
    1) Request-scoped ContextVar (set by server middleware or explicit setters).
    2) ``MLFLOW_WORKSPACE`` environment variable (client-side fallback, including threads).
    """

    if workspace := (_WORKSPACE.get() or "").strip():
        return workspace

    if env_workspace := (MLFLOW_WORKSPACE.get() or "").strip():
        return env_workspace

    return None


def is_request_workspace_resolved() -> bool:
    """Return whether the server resolved the request workspace."""
    return _IS_WORKSPACE_RESOLVED.get()


def _validate_workspace(workspace: str | None) -> None:
    if workspace is not None and workspace != DEFAULT_WORKSPACE_NAME:
        from mlflow.store.workspace.abstract_store import WorkspaceNameValidator

        WorkspaceNameValidator.validate(workspace)


def set_server_request_workspace(workspace: str | None) -> Token[str | None]:
    """
    Server-only setter: bind the workspace to the request ContextVar without touching env.
    """

    _validate_workspace(workspace)
    _IS_WORKSPACE_RESOLVED.set(True)
    return _WORKSPACE.set(workspace)


def set_workspace(workspace: str | None) -> Token[str | None]:
    """
    Client setter: binds the workspace to the current thread and persists to env so child
    threads inherit it.
    """

    _validate_workspace(workspace)
    token = _WORKSPACE.set(workspace)
    if workspace is None:
        MLFLOW_WORKSPACE.unset()
    else:
        MLFLOW_WORKSPACE.set(workspace)
    return token


def clear_server_request_workspace() -> None:
    """Clear the request-scoped ContextVar (does not touch the client env)."""
    _IS_WORKSPACE_RESOLVED.set(False)
    _WORKSPACE.set(None)


class WorkspaceContext:
    """
    Context manager that sets the client workspace (ContextVar + env) for the duration
    of the block. Restores the previous env value on exit.
    """

    def __init__(self, workspace: str | None):
        self._workspace = workspace
        self._token: Token[str | None] | None = None
        self._prev_env_raw: str | None = None

    def __enter__(self) -> str | None:
        self._prev_env_raw = MLFLOW_WORKSPACE.get_raw()
        self._token = set_workspace(self._workspace)
        return self._workspace

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _WORKSPACE.reset(self._token)
            self._token = None

        if self._prev_env_raw is None:
            MLFLOW_WORKSPACE.unset()
        else:
            MLFLOW_WORKSPACE.set(self._prev_env_raw)
        self._prev_env_raw = None

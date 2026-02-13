from __future__ import annotations

from mlflow.environment_variables import MLFLOW_WORKSPACE, MLFLOW_WORKSPACE_STORE_URI

_workspace_store_uri: str | None = None

DEFAULT_WORKSPACE_NAME = "default"
WORKSPACES_DIR_NAME = "workspaces"
WORKSPACE_HEADER_NAME = "X-MLFLOW-WORKSPACE"


def _normalize_workspace(workspace: str | None) -> str | None:
    """Normalize a workspace identifier.

    Args:
        workspace: Raw workspace value, possibly ``None`` or whitespace padded.

    Returns:
        The trimmed workspace string or ``None`` when the input is empty.
    """
    if workspace is None:
        return None
    value = workspace.strip()
    return value or None


def resolve_entity_workspace_name(workspace: str | None) -> str:
    """Determine the workspace to associate with client-facing entities.

    Preference order:
        1. Explicit ``workspace`` argument provided by the backend store.
        2. Active workspace bound via ``mlflow.set_workspace``.
        3. ``MLFLOW_WORKSPACE`` environment variable.
        4. ``DEFAULT_WORKSPACE_NAME``.

    Args:
        workspace: Optional workspace name provided by the store layer.

    Returns:
        A normalized workspace name honoring the preference order.
    """

    if candidate := _normalize_workspace(workspace):
        return candidate

    from mlflow.utils.workspace_context import get_request_workspace

    if candidate := _normalize_workspace(get_request_workspace()):
        return candidate

    if candidate := _normalize_workspace(MLFLOW_WORKSPACE.get()):
        return candidate

    return DEFAULT_WORKSPACE_NAME


def set_workspace_store_uri(uri: str | None) -> None:
    """Set the global workspace provider URI override.

    Args:
        uri: URI of the workspace provider or ``None`` to clear the override.
    """

    global _workspace_store_uri
    _workspace_store_uri = uri
    if uri is None:
        MLFLOW_WORKSPACE_STORE_URI.unset()
    else:
        MLFLOW_WORKSPACE_STORE_URI.set(uri)


def resolve_workspace_store_uri(
    workspace_store_uri: str | None = None, tracking_uri: str | None = None
) -> str | None:
    """Resolve the workspace provider URI according to precedence rules.

    Args:
        workspace_store_uri: URI provided explicitly by the caller.
        tracking_uri: Tracking URI used as the final fallback.

    Returns:
        The workspace provider URI chosen in this order:

        1. Explicit ``workspace_store_uri`` argument.
        2. Value configured via :func:`set_workspace_store_uri`
           or ``MLFLOW_WORKSPACE_STORE_URI``.
        3. The resolved tracking URI.
    """

    if workspace_store_uri is not None:
        return workspace_store_uri

    if configured_uri := get_workspace_store_uri():
        return configured_uri

    # Lazy import to avoid circular dependency during module import.
    from mlflow.tracking._tracking_service import utils as tracking_utils

    return tracking_utils._resolve_tracking_uri(tracking_uri)


def get_workspace_store_uri() -> str | None:
    """Get the current workspace provider URI override, if any.

    Returns:
        The globally configured workspace provider URI, or ``None``.
    """
    return _workspace_store_uri or MLFLOW_WORKSPACE_STORE_URI.get()


__all__ = [
    "DEFAULT_WORKSPACE_NAME",
    "WORKSPACES_DIR_NAME",
    "WORKSPACE_HEADER_NAME",
    "resolve_entity_workspace_name",
    "set_workspace_store_uri",
    "get_workspace_store_uri",
]

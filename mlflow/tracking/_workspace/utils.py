from __future__ import annotations

from mlflow.environment_variables import MLFLOW_WORKSPACE_URI

_workspace_uri: str | None = None


def set_workspace_uri(uri: str | None) -> None:
    """
    Set the global workspace provider URI override.
    """

    global _workspace_uri
    _workspace_uri = uri
    if uri is None:
        MLFLOW_WORKSPACE_URI.unset()
    else:
        MLFLOW_WORKSPACE_URI.set(uri)


def get_workspace_uri() -> str | None:
    """
    Get the current workspace provider URI, if any has been set.
    """

    if _workspace_uri is not None:
        return _workspace_uri
    return MLFLOW_WORKSPACE_URI.get()


def resolve_workspace_uri(
    workspace_uri: str | None = None, tracking_uri: str | None = None
) -> str | None:
    """
    Resolve the workspace provider URI with precedence:

    1. Explicit ``workspace_uri`` argument.
    2. Value configured via :func:`set_workspace_uri` or ``MLFLOW_WORKSPACE_URI``.
    3. The resolved tracking URI.
    """

    if workspace_uri is not None:
        return workspace_uri

    configured_uri = get_workspace_uri()
    if configured_uri is not None:
        return configured_uri

    # Lazy import to avoid circular dependency during module import.
    from mlflow.tracking._tracking_service import utils as tracking_utils

    return tracking_utils._resolve_tracking_uri(tracking_uri)

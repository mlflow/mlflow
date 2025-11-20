"""Workspace-related utility constants and helpers."""

from mlflow.environment_variables import MLFLOW_WORKSPACE

DEFAULT_WORKSPACE_NAME = "default"


def resolve_entity_workspace_name(workspace: str | None) -> str:
    """
    Determine the workspace to associate with client-facing entities.

    Preference order:
      1. Explicit ``workspace`` argument provided by the backend store
      2. Active workspace bound via ``mlflow.set_workspace`` (context var)
      3. ``MLFLOW_WORKSPACE`` environment variable
      4. ``DEFAULT_WORKSPACE_NAME``
    """

    if workspace is not None:
        ws = workspace.strip()
        if ws:
            return ws

    current_workspace = None
    try:
        from mlflow.tracking._workspace.context import get_current_workspace
    except ModuleNotFoundError:  # pragma: no cover - tracking module unavailable for some installs
        current_workspace = None
    else:
        current_workspace = get_current_workspace()

    if current_workspace:
        ws = current_workspace.strip()
        if ws:
            return ws

    env_workspace = MLFLOW_WORKSPACE.get()
    if env_workspace:
        ws = env_workspace.strip()
        if ws:
            return ws

    # Since this is client facing and no workspace was provided, we know DEFAULT_WORKSPACE_NAME is
    # the value to use as this is the default value in the SQL store.
    return DEFAULT_WORKSPACE_NAME


__all__ = ["DEFAULT_WORKSPACE_NAME", "resolve_entity_workspace_name"]

"""
Mixin class providing common workspace functionality for stores.
"""

from __future__ import annotations

from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.utils.workspace_context import get_request_workspace
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME


class WorkspaceAwareMixin:
    """
    Mixin providing common workspace-aware functionality for SQLAlchemy stores.

    Classes using this mixin must have a ManagedSessionMaker attribute.
    """

    @property
    def supports_workspaces(self) -> bool:
        """Indicates this store supports workspace isolation."""
        return True

    @staticmethod
    def _workspaces_enabled() -> bool:
        """Check if workspaces are enabled via environment variable."""
        return MLFLOW_ENABLE_WORKSPACES.get()

    @classmethod
    def _get_active_workspace(cls) -> str:
        """
        Get the active workspace name.

        When workspaces are disabled, returns DEFAULT_WORKSPACE_NAME for backward compatibility.
        When workspaces are enabled, requires an explicit workspace context to be set. Flask and
        FastAPI middlewares resolve (and set) the default workspace whenever the workspace provider
        supports it, so stores can rely on this check to enforce isolation.

        Returns:
            The active workspace name.

        Raises:
            MlflowException: If workspaces are enabled but no workspace context is set.
        """
        if not cls._workspaces_enabled():
            return DEFAULT_WORKSPACE_NAME

        if workspace := get_request_workspace():
            return workspace

        raise MlflowException.invalid_parameter_value(
            "Active workspace is required. Configure a default workspace or call "
            "mlflow.set_workspace() before interacting with the store."
        )

    def _with_workspace_field(self, instance):
        """
        Populate workspace field from active workspace context.
        """
        if hasattr(instance, "workspace"):
            instance.workspace = self._get_active_workspace()
        return instance

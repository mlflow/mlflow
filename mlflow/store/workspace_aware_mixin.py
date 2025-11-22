"""
Mixin class providing common workspace functionality for stores.
"""

from __future__ import annotations

from contextlib import contextmanager

from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.tracking._workspace.context import get_current_workspace
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME


class WorkspaceAwareMixin:
    """
    Mixin providing common workspace-aware functionality for SQLAlchemy stores.

    Classes using this mixin must have a ManagedSessionMaker attribute.
    """

    def supports_workspaces(self) -> bool:
        """Indicates this store supports workspace isolation."""
        return True

    @staticmethod
    def _workspaces_enabled() -> bool:
        """Check if workspaces are enabled via environment variable."""
        return bool(MLFLOW_ENABLE_WORKSPACES.get())

    @classmethod
    def _get_active_workspace(cls) -> str:
        """
        Get the active workspace name.

        When workspaces are disabled, returns DEFAULT_WORKSPACE_NAME for backward compatibility.
        When workspaces are enabled, requires an explicit workspace context to be set.

        Returns:
            The active workspace name.

        Raises:
            MlflowException: If workspaces are enabled but no workspace context is set.
        """
        if not cls._workspaces_enabled():
            return DEFAULT_WORKSPACE_NAME

        # Flask before_request handler sets the workspace context when in web server.
        # For CLI/script usage, this may be set via mlflow.set_workspace().
        workspace = get_current_workspace()
        if workspace:
            return workspace

        raise MlflowException.invalid_parameter_value(
            "Active workspace is required. Configure a default workspace or call "
            "mlflow.set_workspace() before interacting with the store."
        )

    @contextmanager
    def _workspace_session(self):
        """
        Context manager that provides both database session and active workspace.

        This helper reduces repetitive calls to _get_active_workspace() followed by
        ManagedSessionMaker by combining them into a single context manager.

        Yields:
            tuple: (session, workspace) where session is a SQLAlchemy session and
                   workspace is the active workspace name.

        Example:
            with self._workspace_session() as (session, workspace):
                experiment = session.query(SqlExperiment).filter(
                    SqlExperiment.experiment_id == experiment_id,
                    SqlExperiment.workspace == workspace
                ).one_or_none()
        """
        workspace = self._get_active_workspace()
        with self.ManagedSessionMaker() as session:
            yield session, workspace

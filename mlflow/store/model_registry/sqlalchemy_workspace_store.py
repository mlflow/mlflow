"""
Workspace-aware variant of the model registry SQLAlchemy store.
"""

from __future__ import annotations

import logging

from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.workspace_aware_mixin import WorkspaceAwareMixin

_logger = logging.getLogger(__name__)


class WorkspaceAwareSqlAlchemyStore(WorkspaceAwareMixin, SqlAlchemyStore):
    """
    Workspace-aware variant of the model registry SQLAlchemy store.

    Workspace filtering is handled by the base ``SqlAlchemyStore`` via
    ``_get_active_workspace()``, which ``WorkspaceAwareMixin`` overrides
    to return the workspace from the current request context.
    """

    def __init__(self, db_uri):
        super().__init__(db_uri)

    def _initialize_store_state(self):
        """
        Initialize store state for workspace-aware mode.

        In workspace mode, we skip the non-default workspace validation since
        having entries in different workspaces is expected and correct behavior.
        """
        # No validation needed in workspace-aware mode - entries in different
        # workspaces are expected and correct behavior

"""
Workspace-aware variant of the model registry SQLAlchemy store.
"""

from __future__ import annotations

import logging

from mlflow.store.db.workspace_isolated_model import WorkspaceIsolatedModel
from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.workspace_aware_mixin import WorkspaceAwareMixin

_logger = logging.getLogger(__name__)


class WorkspaceAwareSqlAlchemyStore(WorkspaceAwareMixin, SqlAlchemyStore):
    """
    Workspace-aware variant of the model registry SQLAlchemy store.

    This store adds workspace filtering to all queries, ensuring data isolation
    between workspaces.
    """

    def __init__(self, db_uri):
        super().__init__(db_uri)

    def _get_query(self, session, model):
        query = super()._get_query(session, model)
        if issubclass(model, WorkspaceIsolatedModel):
            workspace = self._get_active_workspace()
            return model.workspace_query_filter(query, session, workspace)
        return query

    def _initialize_store_state(self):
        """
        Initialize store state for workspace-aware mode.

        In workspace mode, we skip the non-default workspace validation since
        having entries in different workspaces is expected and correct behavior.
        """
        # No validation needed in workspace-aware mode - entries in different
        # workspaces are expected and correct behavior

    def _get_workspace_clauses(self, model):
        if issubclass(model, WorkspaceIsolatedModel):
            workspace = self._get_active_workspace()
            return [model.workspace == workspace]
        return []

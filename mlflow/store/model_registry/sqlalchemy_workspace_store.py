"""
Workspace-aware variant of the model registry SQLAlchemy store.
"""

from __future__ import annotations

import logging

from mlflow.store.model_registry.dbmodels.models import (
    SqlModelVersion,
    SqlModelVersionTag,
    SqlRegisteredModel,
    SqlRegisteredModelAlias,
    SqlRegisteredModelTag,
    SqlWebhook,
)
from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.workspace_aware_mixin import WorkspaceAwareMixin

_logger = logging.getLogger(__name__)

_WORKSPACE_ISOLATED_MODELS = (
    SqlRegisteredModel,
    SqlModelVersion,
    SqlWebhook,
    SqlRegisteredModelTag,
    SqlModelVersionTag,
    SqlRegisteredModelAlias,
)


class WorkspaceAwareSqlAlchemyStore(WorkspaceAwareMixin, SqlAlchemyStore):
    """
    Workspace-aware variant of the model registry SQLAlchemy store.

    This store adds workspace filtering to all queries, ensuring data isolation
    between workspaces.
    """

    def __init__(self, db_uri):
        super().__init__(db_uri)

    def _get_query(self, session, model):
        """
        Return a query for ``model`` filtered by the active workspace.
        """
        query = super()._get_query(session, model)
        workspace = self._get_active_workspace()

        if model in _WORKSPACE_ISOLATED_MODELS:
            return query.filter(model.workspace == workspace)

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
        """
        Return workspace filter clauses for the model.
        """
        workspace = self._get_active_workspace()

        if model in _WORKSPACE_ISOLATED_MODELS:
            return [model.workspace == workspace]

        return []

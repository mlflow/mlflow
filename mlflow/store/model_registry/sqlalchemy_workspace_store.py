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

        if model is SqlRegisteredModel:
            return query.filter(SqlRegisteredModel.workspace == workspace)

        if model is SqlModelVersion:
            return query.filter(SqlModelVersion.workspace == workspace)

        if model is SqlWebhook:
            return query.filter(SqlWebhook.workspace == workspace)

        if model is SqlRegisteredModelTag:
            return query.filter(SqlRegisteredModelTag.workspace == workspace)

        if model is SqlModelVersionTag:
            return query.filter(SqlModelVersionTag.workspace == workspace)

        if model is SqlRegisteredModelAlias:
            return query.filter(SqlRegisteredModelAlias.workspace == workspace)

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

        if model is SqlRegisteredModel:
            return [SqlRegisteredModel.workspace == workspace]

        if model is SqlModelVersion:
            return [SqlModelVersion.workspace == workspace]

        if model is SqlWebhook:
            return [SqlWebhook.workspace == workspace]

        if model is SqlRegisteredModelTag:
            return [SqlRegisteredModelTag.workspace == workspace]

        if model is SqlModelVersionTag:
            return [SqlModelVersionTag.workspace == workspace]

        if model is SqlRegisteredModelAlias:
            return [SqlRegisteredModelAlias.workspace == workspace]

        return []

"""
Workspace-aware variant of the jobs SQLAlchemy store.
"""

from __future__ import annotations

import logging

from mlflow.store.db.workspace_isolated_model import WorkspaceIsolatedModel
from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore
from mlflow.store.workspace_aware_mixin import WorkspaceAwareMixin

_logger = logging.getLogger(__name__)


class WorkspaceAwareSqlAlchemyJobStore(WorkspaceAwareMixin, SqlAlchemyJobStore):
    """
    Workspace-aware variant of the jobs SQLAlchemy store.

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

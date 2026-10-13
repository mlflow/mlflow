"""
Workspace-aware variant of the jobs SQLAlchemy store.
"""

from __future__ import annotations

import logging

from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore
from mlflow.store.tracking.dbmodels.models import SqlJob
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
        """
        Return a query for ``model`` filtered by the active workspace.
        """
        query = super()._get_query(session, model)
        workspace = self._get_active_workspace()

        if model is SqlJob:
            return query.filter(SqlJob.workspace == workspace)

        return query

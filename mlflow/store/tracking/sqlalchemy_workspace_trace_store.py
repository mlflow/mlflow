from __future__ import annotations

import logging

from sqlalchemy.future import select

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.tracking.dbmodels.models import (
    SqlAssessments,
    SqlExperiment,
    SqlTraceInfo,
)

_logger = logging.getLogger(__name__)


from mlflow.store.tracking.sqlalchemy_trace_store import SqlAlchemyTraceStore


class WorkspaceAwareSqlAlchemyTraceStore(SqlAlchemyTraceStore):
    def _trace_query(self, session, for_update_or_delete=False, workspace=None):
        """
        Return a workspace-scoped trace query.

        Both plain reads and locking reads target `trace_info` directly via
        `session.query(SqlTraceInfo)` and scope results with an
        `experiment_id IN (...)` subquery over experiments in the selected workspace. This keeps
        the query shape anchored on trace rows rather than relying on a workspace-aware join
        through experiments, which is especially important when applying row locks so the lock
        lands directly on `trace_info` rows. Callers may pass an explicit workspace snapshot when
        a multi-step write needs stable scoping across several queries.
        """
        workspace = workspace or self._tracking_store._get_active_workspace()
        workspace_experiment_ids = (
            session
            .query(SqlExperiment.experiment_id)
            .filter(SqlExperiment.workspace == workspace)
            .subquery()
        )
        query = session.query(SqlTraceInfo).filter(
            SqlTraceInfo.experiment_id.in_(select(workspace_experiment_ids.c.experiment_id))
        )
        if for_update_or_delete:
            return self._apply_trace_row_lock(query)
        return query

    def _validate_trace_accessible(self, session, trace_id: str) -> None:
        workspace = self._tracking_store._get_active_workspace()
        exists_row = (
            session
            .query(SqlTraceInfo.request_id)
            .filter(SqlTraceInfo.request_id == trace_id)
            .filter(
                SqlTraceInfo.experiment_id.in_(
                    session.query(SqlExperiment.experiment_id).filter(
                        SqlExperiment.workspace == workspace
                    )
                )
            )
            .first()
        )
        if exists_row is None:
            raise MlflowException(
                f"Trace with ID '{trace_id}' not found.",
                RESOURCE_DOES_NOT_EXIST,
            )

    def _get_sql_assessment(self, session, trace_id: str, assessment_id: str) -> SqlAssessments:
        trace_subquery = (
            self
            ._trace_query(session)
            .with_entities(SqlTraceInfo.request_id)
            .filter(SqlTraceInfo.request_id == trace_id)
            .subquery()
        )

        sql_assessment = (
            session
            .query(SqlAssessments)
            .join(trace_subquery, SqlAssessments.trace_id == trace_subquery.c.request_id)
            .filter(SqlAssessments.assessment_id == assessment_id)
            .one_or_none()
        )
        if sql_assessment is None:
            trace_record = (
                self._trace_query(session).filter(SqlTraceInfo.request_id == trace_id).one_or_none()
            )
            if trace_record is None:
                raise MlflowException(
                    f"Trace with ID '{trace_id}' not found.",
                    RESOURCE_DOES_NOT_EXIST,
                )

            raise MlflowException(
                f"Assessment with ID '{assessment_id}' not found for trace '{trace_id}'",
                RESOURCE_DOES_NOT_EXIST,
            )
        return sql_assessment

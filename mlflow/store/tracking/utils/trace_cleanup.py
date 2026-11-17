from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from mlflow.entities.entity_type import EntityAssociationType
from mlflow.store.tracking.dbmodels.models import (
    SqlAssessmentDailyRollup,
    SqlEntityAssociation,
    SqlRun,
    SqlSpanCostDailyRollup,
    SqlTraceInfo,
    SqlTraceMetricDailyRollup,
    SqlTraceRollupRebuild,
)


def delete_trace_associations(session: Session, trace_ids: list[str]) -> None:
    if trace_ids:
        session.query(SqlEntityAssociation).filter(
            SqlEntityAssociation.source_type == EntityAssociationType.TRACE,
            SqlEntityAssociation.source_id.in_(trace_ids),
        ).delete(synchronize_session=False)


def delete_experiment_trace_dependencies(session: Session, experiment_ids: list[int]) -> None:
    """Remove non-cascading SQL trace data before its experiment ownership is lost."""
    traces = session.query(SqlTraceInfo.request_id).filter(
        SqlTraceInfo.experiment_id.in_(experiment_ids)
    )
    runs = session.query(SqlRun.run_uuid).filter(SqlRun.experiment_id.in_(experiment_ids))
    session.query(SqlEntityAssociation).filter(
        SqlEntityAssociation.source_type == EntityAssociationType.TRACE,
        or_(
            SqlEntityAssociation.source_id.in_(traces),
            and_(
                SqlEntityAssociation.destination_type == EntityAssociationType.RUN,
                SqlEntityAssociation.destination_id.in_(runs),
            ),
        ),
    ).delete(synchronize_session=False)
    for model in (
        SqlTraceRollupRebuild,
        SqlTraceMetricDailyRollup,
        SqlSpanCostDailyRollup,
        SqlAssessmentDailyRollup,
    ):
        session.query(model).filter(model.experiment_id.in_(experiment_ids)).delete(
            synchronize_session=False
        )

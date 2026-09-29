from sqlalchemy.orm.query import Query

from mlflow.store.tracking.dbmodels.models import SqlSpan, SqlTraceInfo


def _apply_postgres_trace_first_span_query(query: Query) -> Query:
    """Materialize filtered trace IDs before joining spans on PostgreSQL."""
    metric_trace_ids = (
        query
        .with_entities(SqlTraceInfo.request_id.label("trace_id"))
        .cte("metric_trace_ids")
        .prefix_with("MATERIALIZED")
    )
    return query.session.query(SqlSpan).join(
        metric_trace_ids, SqlSpan.trace_id == metric_trace_ids.c.trace_id
    )

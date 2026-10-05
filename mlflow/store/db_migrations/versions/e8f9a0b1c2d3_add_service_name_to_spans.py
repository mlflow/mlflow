"""add service name to spans

Revision ID: e8f9a0b1c2d3
Revises: dc11669786a5
Create Date: 2026-10-04 00:00:00.000000

Historical span content does not include OTLP resource attributes, so the backfill can only
recover service.name when it was also recorded as a span attribute. Resource-only values remain
NULL.

"""

import json
import logging

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mssql import NVARCHAR

revision = "e8f9a0b1c2d3"
down_revision = "dc11669786a5"
branch_labels = None
depends_on = None

_logger = logging.getLogger(__name__)

_BATCH_SIZE = 250
_SERVICE_NAME_KEY = "service.name"


def _service_name_from_span_content(content):
    try:
        span = json.loads(content)
    except (TypeError, ValueError):
        return None

    attributes = span.get("attributes") if isinstance(span, dict) else None
    if not isinstance(attributes, dict) or _SERVICE_NAME_KEY not in attributes:
        return None

    value = attributes[_SERVICE_NAME_KEY]
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            pass
    return None if value is None else str(value)


def _backfill_service_names():
    bind = op.get_bind()
    metadata = sa.MetaData()
    spans = sa.Table("spans", metadata, autoload_with=bind)
    update_stmt = (
        spans
        .update()
        .where(
            spans.c.trace_id == sa.bindparam("trace_id_param"),
            spans.c.span_id == sa.bindparam("span_id_param"),
        )
        .values(service_name=sa.bindparam("service_name_param"))
    )

    last_span_key = None
    while True:
        page_stmt = (
            sa
            .select(
                spans.c.trace_id,
                spans.c.span_id,
                spans.c.content,
            )
            .where(spans.c.service_name.is_(None))
            .order_by(spans.c.trace_id, spans.c.span_id)
        )
        if last_span_key is not None:
            last_trace_id, last_span_id = last_span_key
            page_stmt = page_stmt.where(
                sa.or_(
                    spans.c.trace_id > last_trace_id,
                    sa.and_(
                        spans.c.trace_id == last_trace_id,
                        spans.c.span_id > last_span_id,
                    ),
                )
            )
        batch = bind.execute(page_stmt.limit(_BATCH_SIZE)).all()
        if not batch:
            break

        updates = []
        for row in batch:
            service_name = _service_name_from_span_content(row.content)
            if service_name is not None:
                updates.append({
                    "trace_id_param": row.trace_id,
                    "span_id_param": row.span_id,
                    "service_name_param": service_name,
                })
        if updates:
            _execute_backfill_updates(bind, update_stmt, updates)
        last_span_key = (batch[-1].trace_id, batch[-1].span_id)


def _execute_backfill_updates(bind, update_stmt, updates):
    result = bind.execute(update_stmt, updates)
    if result.supports_sane_multi_rowcount() and result.rowcount != len(updates):
        _logger.warning(
            "Span service-name backfill updated %s of %s rows; unmatched rows will keep a NULL "
            "service_name",
            result.rowcount,
            len(updates),
        )


def upgrade():
    op.add_column(
        "spans",
        sa.Column(
            "service_name",
            sa.Text().with_variant(NVARCHAR(None), "mssql"),
            nullable=True,
        ),
    )
    _backfill_service_names()


def downgrade():
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table("spans") as batch_op:
            # SQLite copies reflected columns during batch recreation but rejects explicit values
            # for stored generated columns. Recreate duration_ns so SQLite recomputes it.
            batch_op.drop_column("duration_ns")
            batch_op.drop_column("service_name")
            batch_op.add_column(
                sa.Column(
                    "duration_ns",
                    sa.BigInteger(),
                    sa.Computed(
                        "end_time_unix_nano - start_time_unix_nano",
                        persisted=True,
                    ),
                    nullable=True,
                ),
                insert_before="content",
            )
    else:
        op.drop_column("spans", "service_name")

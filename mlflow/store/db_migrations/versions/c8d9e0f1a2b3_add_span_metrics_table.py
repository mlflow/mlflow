"""add span metrics table

Create Date: 2026-01-27 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql

# revision identifiers, used by Alembic.
revision = "c8d9e0f1a2b3"
down_revision = "d0e1f2a3b4c5"
branch_labels = None
depends_on = None


def _get_json_type():
    """Get appropriate JSON type for the current database."""
    dialect_name = op.get_bind().dialect.name
    if dialect_name == "mssql":
        return mssql.JSON
    else:
        return sa.JSON


def upgrade():
    json_type = _get_json_type()

    op.create_table(
        "span_metrics",
        sa.Column("trace_id", sa.String(length=50), nullable=False),
        sa.Column("span_id", sa.String(length=50), nullable=False),
        sa.Column("key", sa.String(length=250), nullable=False),
        sa.Column("value", sa.Float(precision=53), nullable=True),
        sa.Column("metric_metadata", json_type, nullable=True),
        sa.ForeignKeyConstraint(
            ["trace_id", "span_id"],
            ["spans.trace_id", "spans.span_id"],
            name="fk_span_metrics_span",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("trace_id", "span_id", "key", name="span_metrics_pk"),
    )

    with op.batch_alter_table("span_metrics", schema=None) as batch_op:
        batch_op.create_index(
            "index_span_metrics_trace_id_span_id",
            ["trace_id", "span_id"],
            unique=False,
        )


def downgrade():
    op.drop_table("span_metrics")

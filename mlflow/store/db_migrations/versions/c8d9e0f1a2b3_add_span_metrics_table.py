"""add span metrics and span attributes tables

Create Date: 2026-01-27 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "c8d9e0f1a2b3"
down_revision = "d3e4f5a6b7c8"
branch_labels = None
depends_on = None


def upgrade():
    # Create span_metrics table
    op.create_table(
        "span_metrics",
        sa.Column("trace_id", sa.String(length=50), nullable=False),
        sa.Column("span_id", sa.String(length=50), nullable=False),
        sa.Column("key", sa.String(length=250), nullable=False),
        sa.Column("value", sa.Float(precision=53), nullable=True),
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

    # Create span_attributes table
    op.create_table(
        "span_attributes",
        sa.Column("trace_id", sa.String(length=50), nullable=False),
        sa.Column("span_id", sa.String(length=50), nullable=False),
        sa.Column("key", sa.String(length=250), nullable=False),
        sa.Column("value", sa.String(length=8000), nullable=True),
        sa.ForeignKeyConstraint(
            ["trace_id", "span_id"],
            ["spans.trace_id", "spans.span_id"],
            name="fk_span_attributes_span",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("trace_id", "span_id", "key", name="span_attributes_pk"),
    )

    with op.batch_alter_table("span_attributes", schema=None) as batch_op:
        batch_op.create_index(
            "index_span_attributes_trace_id_span_id",
            ["trace_id", "span_id"],
            unique=False,
        )
        batch_op.create_index(
            "index_span_attributes_key",
            ["key"],
            unique=False,
        )


def downgrade():
    op.drop_table("span_attributes")
    op.drop_table("span_metrics")

"""add trace metrics table

Create Date: 2025-12-04 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

from mlflow.store.tracking.dbmodels.models import SqlTraceMetrics

# revision identifiers, used by Alembic.
revision = "b7c8d9e0f1a2"
down_revision = "1bd49d398cd23"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        SqlTraceMetrics.__tablename__,
        sa.Column("request_id", sa.String(length=50), nullable=False),
        sa.Column("key", sa.String(length=250), nullable=False),
        sa.Column("value", sa.Float(precision=53), nullable=True),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["trace_info.request_id"],
            name="fk_trace_metrics_request_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("request_id", "key", name="trace_metrics_pk"),
    )

    # Add index on request_id for faster lookups
    with op.batch_alter_table(SqlTraceMetrics.__tablename__, schema=None) as batch_op:
        batch_op.create_index(
            f"index_{SqlTraceMetrics.__tablename__}_request_id",
            ["request_id"],
            unique=False,
        )


def downgrade():
    op.drop_table(SqlTraceMetrics.__tablename__)

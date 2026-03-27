"""add trace_views table

Create Date: 2026-03-26 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

from mlflow.store.tracking.dbmodels.models import SqlTraceView

# revision identifiers, used by Alembic.
revision = "eb885a9619f6"
down_revision = "c3d6457b6d8a"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        SqlTraceView.__tablename__,
        sa.Column("view_id", sa.String(length=50), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("trace_id", sa.String(length=50), nullable=True),
        sa.Column("experiment_id", sa.Integer(), nullable=True),
        sa.Column("span_filter", sa.Text(), nullable=True),
        sa.Column("input_path", sa.Text(), nullable=True),
        sa.Column("output_path", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(length=256), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_timestamp", sa.BigInteger(), nullable=False),
        sa.Column("last_updated_timestamp", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["trace_id"],
            ["trace_info.request_id"],
            name="fk_trace_views_trace_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_trace_views_experiment_id",
        ),
        sa.PrimaryKeyConstraint("view_id", name="trace_views_pk"),
        sa.CheckConstraint(
            "(trace_id IS NOT NULL AND experiment_id IS NULL) OR "
            "(trace_id IS NULL AND experiment_id IS NOT NULL)",
            name="ck_trace_views_scope",
        ),
    )

    with op.batch_alter_table(SqlTraceView.__tablename__, schema=None) as batch_op:
        batch_op.create_index(
            f"index_{SqlTraceView.__tablename__}_trace_id_created_timestamp",
            ["trace_id", "created_timestamp"],
            unique=False,
        )
        batch_op.create_index(
            f"index_{SqlTraceView.__tablename__}_experiment_id_created_timestamp",
            ["experiment_id", "created_timestamp"],
            unique=False,
        )
        batch_op.create_index(
            f"index_{SqlTraceView.__tablename__}_last_updated_timestamp",
            ["last_updated_timestamp"],
            unique=False,
        )


def downgrade():
    op.drop_table(SqlTraceView.__tablename__)

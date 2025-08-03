"""add spans table

Revision ID: a1b2c3d4e5f6
Revises: 770bee3ae1dd
Create Date: 2025-08-03 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

from mlflow.store.tracking.dbmodels.models import SqlSpan

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = "770bee3ae1dd"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        SqlSpan.__tablename__,
        sa.Column("trace_id", sa.String(length=50), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("span_id", sa.String(length=50), nullable=False),
        sa.Column("parent_span_id", sa.String(length=50), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("start_time_unix_nano", sa.BigInteger(), nullable=False),
        sa.Column("end_time_unix_nano", sa.BigInteger(), nullable=True),
        sa.Column("trace_state", sa.Text(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["trace_id"],
            ["trace_info.request_id"],
            name="fk_spans_trace_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_spans_experiment_id",
        ),
        sa.PrimaryKeyConstraint("trace_id", "span_id", name="spans_pk"),
    )

    with op.batch_alter_table(SqlSpan.__tablename__, schema=None) as batch_op:
        batch_op.create_index(
            f"index_{SqlSpan.__tablename__}_trace_id",
            ["trace_id"],
            unique=False,
        )
        batch_op.create_index(
            f"index_{SqlSpan.__tablename__}_experiment_id",
            ["experiment_id"],
            unique=False,
        )
        batch_op.create_index(
            f"index_{SqlSpan.__tablename__}_experiment_id_start_time",
            ["experiment_id", "start_time_unix_nano"],
            unique=False,
        )


def downgrade():
    op.drop_table(SqlSpan.__tablename__)

"""add labeling sessions table

Create Date: 2025-09-25 00:12:14.068102

"""

import time

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "73ab2d98a1ec"
down_revision = "bf29a5ff90ea"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "labeling_sessions",
        sa.Column("labeling_session_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=500), nullable=False),
        sa.Column("mlflow_run_id", sa.String(length=32), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("creation_time", sa.BigInteger(), default=lambda: int(time.time() * 1000)),
        sa.Column("last_updated_time", sa.BigInteger(), default=lambda: int(time.time() * 1000)),
        sa.PrimaryKeyConstraint("labeling_session_id", name="labeling_sessions_pk"),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="labeling_sessions_experiment_id_fkey",
            onupdate="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["mlflow_run_id"],
            ["runs.run_uuid"],
            name="labeling_sessions_mlflow_run_id_fkey",
            onupdate="CASCADE",
        ),
    )
    with op.batch_alter_table("labeling_sessions", schema=None) as batch_op:
        batch_op.create_index(
            "index_labeling_sessions_experiment_id",
            ["experiment_id"],
            unique=False,
        )
        batch_op.create_index(
            "index_labeling_sessions_mlflow_run_id",
            ["mlflow_run_id"],
            unique=False,
        )
        batch_op.create_index(
            "index_labeling_sessions_creation_time",
            ["creation_time"],
            unique=False,
        )


def downgrade():
    op.drop_table("labeling_sessions")

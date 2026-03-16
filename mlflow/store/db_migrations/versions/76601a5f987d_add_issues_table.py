"""add issues table

Create Date: 2026-02-26 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

from mlflow.store.tracking.dbmodels.models import SqlIssue

# revision identifiers, used by Alembic.
revision = "76601a5f987d"
down_revision = "e1f2a3b4c5d6"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        SqlIssue.__tablename__,
        sa.Column("issue_id", sa.String(length=36), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=250), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("severity", sa.String(length=50), nullable=True),
        sa.Column("root_causes", sa.Text(), nullable=True),
        sa.Column("source_run_id", sa.String(length=32), nullable=True),
        sa.Column("categories", sa.Text(), nullable=True),
        sa.Column("created_timestamp", sa.BigInteger(), nullable=False),
        sa.Column("last_updated_timestamp", sa.BigInteger(), nullable=False),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_issues_experiment_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_run_id"],
            ["runs.run_uuid"],
            name="fk_issues_source_run_id",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("issue_id", name="issues_pk"),
    )

    with op.batch_alter_table(SqlIssue.__tablename__, schema=None) as batch_op:
        batch_op.create_index(
            f"index_{SqlIssue.__tablename__}_experiment_id",
            ["experiment_id"],
            unique=False,
        )
        batch_op.create_index(
            f"index_{SqlIssue.__tablename__}_source_run_id",
            ["source_run_id"],
            unique=False,
        )
        batch_op.create_index(
            f"index_{SqlIssue.__tablename__}_status",
            ["status"],
            unique=False,
        )


def downgrade():
    op.drop_table(SqlIssue.__tablename__)

"""add scorer table

Revision ID: 534353b11cbc
Revises: 770bee3ae1dd
Create Date: 2025-01-27 10:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

from mlflow.store.tracking.dbmodels.models import SqlScorer

# revision identifiers, used by Alembic.
revision = "534353b11cbc"
down_revision = "770bee3ae1dd"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        SqlScorer.__tablename__,
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("scorer_name", sa.String(length=256), nullable=False),
        sa.Column("scorer_version", sa.Integer(), nullable=False),
        sa.Column("serialized_scorer", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_scorers_experiment_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "experiment_id", "scorer_name", "scorer_version", name="scorer_pk"
        ),
    )

    with op.batch_alter_table(SqlScorer.__tablename__, schema=None) as batch_op:
        batch_op.create_index(
            f"index_{SqlScorer.__tablename__}_experiment_id",
            ["experiment_id"],
            unique=False,
        )
        batch_op.create_index(
            f"index_{SqlScorer.__tablename__}_experiment_id_scorer_name",
            ["experiment_id", "scorer_name"],
            unique=False,
        )


def downgrade():
    op.drop_table(SqlScorer.__tablename__)

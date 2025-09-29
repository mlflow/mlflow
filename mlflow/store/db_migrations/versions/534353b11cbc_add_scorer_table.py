"""add scorer tables

Create Date: 2025-01-27 10:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

from mlflow.store.tracking.dbmodels.models import SqlScorer, SqlScorerVersion

# revision identifiers, used by Alembic.
revision = "534353b11cbc"
down_revision = "1a0cddfcaa16"
branch_labels = None
depends_on = None


def upgrade():
    # Create the scorers table (experiment_id, scorer_name, scorer_id)
    op.create_table(
        SqlScorer.__tablename__,
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("scorer_name", sa.String(length=256), nullable=False),
        sa.Column("scorer_id", sa.String(length=36), nullable=False),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_scorers_experiment_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("scorer_id", name="scorer_pk"),
    )

    # Create the scorer_versions table (scorer_id, scorer_version, serialized_scorer, creation_time)
    op.create_table(
        SqlScorerVersion.__tablename__,
        sa.Column("scorer_id", sa.String(length=36), nullable=False),
        sa.Column("scorer_version", sa.Integer(), nullable=False),
        sa.Column("serialized_scorer", sa.Text(), nullable=False),
        sa.Column("creation_time", sa.BigInteger(), nullable=True),
        sa.ForeignKeyConstraint(
            ["scorer_id"],
            ["scorers.scorer_id"],
            name="fk_scorer_versions_scorer_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("scorer_id", "scorer_version", name="scorer_version_pk"),
    )

    # Create indexes
    with op.batch_alter_table(SqlScorer.__tablename__, schema=None) as batch_op:
        batch_op.create_index(
            f"index_{SqlScorer.__tablename__}_experiment_id_scorer_name",
            ["experiment_id", "scorer_name"],
            unique=True,
        )

    with op.batch_alter_table(SqlScorerVersion.__tablename__, schema=None) as batch_op:
        batch_op.create_index(
            f"index_{SqlScorerVersion.__tablename__}_scorer_id",
            ["scorer_id"],
            unique=False,
        )


def downgrade():
    op.drop_table(SqlScorerVersion.__tablename__)
    op.drop_table(SqlScorer.__tablename__)

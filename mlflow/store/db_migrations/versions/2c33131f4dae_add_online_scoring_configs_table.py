"""add online_scoring_configs table

Create Date: 2025-01-27 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

from mlflow.store.tracking.dbmodels.models import SqlOnlineScoringConfig

# revision identifiers, used by Alembic.
revision = "2c33131f4dae"
down_revision = "c9d4e5f6a7b8"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        SqlOnlineScoringConfig.__tablename__,
        sa.Column("online_scoring_config_id", sa.String(length=36), nullable=False),
        sa.Column("scorer_id", sa.String(length=36), nullable=False),
        sa.Column("sample_rate", sa.types.Float(precision=53), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("filter_string", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["scorer_id"],
            ["scorers.scorer_id"],
            name="fk_online_scoring_configs_scorer_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_online_scoring_configs_experiment_id",
        ),
        sa.PrimaryKeyConstraint("online_scoring_config_id", name="online_scoring_config_pk"),
    )


def downgrade():
    op.drop_table(SqlOnlineScoringConfig.__tablename__)

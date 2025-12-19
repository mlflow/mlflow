"""Add Experiment Tags Table

Create Date: 2019-07-15 17:46:42.704214

"""

import sqlalchemy as sa
from alembic import op

from mlflow.store.tracking.dbmodels.models import SqlExperimentTag

# revision identifiers, used by Alembic.
revision = "df50e92ffc5e"
down_revision = "181f10493468"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        SqlExperimentTag.__tablename__,
        sa.Column("key", sa.String(length=250), primary_key=True, nullable=False),
        sa.Column("value", sa.String(length=5000)),
        sa.Column(
            "experiment_id",
            sa.Integer(),
            sa.ForeignKey("experiments.experiment_id"),
            primary_key=True,
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("key", "experiment_id", name="experiment_tag_pk"),
    )


def downgrade():
    pass

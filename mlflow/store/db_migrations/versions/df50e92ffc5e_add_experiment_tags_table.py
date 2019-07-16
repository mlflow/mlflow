"""Add Experiment Tags Table

Revision ID: df50e92ffc5e
Revises: 90e64c465722
Create Date: 2019-07-15 17:46:42.704214

"""
from alembic import op
import sqlalchemy as sa
from mlflow.store.dbmodels.models import SqlExperimentTag

# revision identifiers, used by Alembic.
revision = 'df50e92ffc5e'
down_revision = '90e64c465722'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(SqlExperimentTag.__tablename__,
        *SqlExperimentTag.__table__.columns
    )


def downgrade():
    pass

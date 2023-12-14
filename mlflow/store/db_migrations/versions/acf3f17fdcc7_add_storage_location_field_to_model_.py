"""add storage location field to model versions

Revision ID: acf3f17fdcc7
Revises: 2d6e25af4d3e
Create Date: 2023-10-23 15:26:53.062080

"""
from alembic import op
import sqlalchemy as sa
from mlflow.store.model_registry.dbmodels.models import SqlModelVersion


# revision identifiers, used by Alembic.
revision = "acf3f17fdcc7"
down_revision = "2d6e25af4d3e"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        SqlModelVersion.__tablename__,
        sa.Column("storage_location", sa.String(500), nullable=True, default=None),
    )


def downgrade():
    pass

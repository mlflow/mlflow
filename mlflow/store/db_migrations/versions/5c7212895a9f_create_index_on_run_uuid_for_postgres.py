"""create index on run_uuid for postgres

Revision ID: 5c7212895a9f
Revises: beed69c8e3d1
Create Date: 2022-03-03 09:49:26.082301

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "5c7212895a9f"
down_revision = "beed69c8e3d1"
branch_labels = None
depends_on = None


def upgrade():
    # This part of the migration is only relevant for PostgreSQL.
    # As a fix for https://github.com/mlflow/mlflow/issues/3785, create indexes on run_uuid columns
    # to speed up SQL operations.
    bind = op.get_bind()
    if bind.engine.name == "postgresql":
        for table in ["params", "metrics", "latest_metrics", "tags"]:
            op.create_index(f"index_{table}_run_uuid", table, ["run_uuid"])


def downgrade():
    pass

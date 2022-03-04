"""create index on run_uuid

Revision ID: bd07f7e963c5
Revises: c48cb773bb87
Create Date: 2022-03-03 10:14:34.037978

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "bd07f7e963c5"
down_revision = "c48cb773bb87"
branch_labels = None
depends_on = None


def upgrade():
    # As a fix for https://github.com/mlflow/mlflow/issues/3785, create an index on run_uuid columns
    # that have a foreign key constraint to speed up SQL operations.
    for table in ["params", "metrics", "latest_metrics", "tags"]:
        op.create_index(f"index_{table}_run_uuid", table, ["run_uuid"])


def downgrade():
    pass

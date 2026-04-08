"""Add index on model_version_tags (workspace, name, version)

The workspace migration (1b5f0d9ad7c1) changed the primary key of
model_version_tags to (workspace, key, name, version).  SQLAlchemy's
relationship loader joins on (workspace, name, version) to fetch tags
for a batch of model versions, but the PK has ``key`` between
``workspace`` and ``name``, so the join can only use the first PK
column.  This index provides an efficient lookup path for that join.

Create Date: 2026-04-08 00:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "7d34483879f0"
down_revision = "ae8bbe7743c9"
branch_labels = None
depends_on = None


def upgrade():
    op.create_index(
        "idx_model_version_tags_workspace_name_version",
        "model_version_tags",
        ["workspace", "name", "version"],
    )


def downgrade():
    op.drop_index(
        "idx_model_version_tags_workspace_name_version",
        table_name="model_version_tags",
    )

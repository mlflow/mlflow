"""Add indexes for relationship loading after workspace PK migration

The workspace migration (1b5f0d9ad7c1) changed primary keys to include
``workspace`` as the leading column.  SQLAlchemy's relationship loader
joins on (workspace, name, version) for model_version_tags and
(workspace, name) for registered_model_tags, but the PKs have ``key``
between ``workspace`` and ``name``, so the join can only use the first
PK column.  These indexes provide efficient lookup paths for those joins.

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
    op.create_index(
        "idx_registered_model_tags_workspace_name",
        "registered_model_tags",
        ["workspace", "name"],
    )


def downgrade():
    op.drop_index(
        "idx_registered_model_tags_workspace_name",
        table_name="registered_model_tags",
    )
    op.drop_index(
        "idx_model_version_tags_workspace_name_version",
        table_name="model_version_tags",
    )

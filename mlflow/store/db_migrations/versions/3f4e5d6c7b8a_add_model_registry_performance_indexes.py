"""add model registry performance indexes

Create Date: 2026-01-10 02:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "3f4e5d6c7b8a"
down_revision = "2c33131f4dae"
branch_labels = None
depends_on = None


def upgrade():
    # Add index on model_versions for efficient latest version queries
    # This supports the query: SELECT * FROM model_versions WHERE name IN (...) AND current_stage != '...' GROUP BY name, current_stage
    op.create_index(
        "idx_model_versions_name_stage_version",
        "model_versions",
        ["name", "current_stage", "version"],
        unique=False,
    )

    # Add index on model_version_tags for efficient tag lookups by model version
    # This supports the query: SELECT * FROM model_version_tags WHERE name = ? AND version = ?
    op.create_index(
        "idx_model_version_tags_name_version",
        "model_version_tags",
        ["name", "version"],
        unique=False,
    )


def downgrade():
    op.drop_index("idx_model_version_tags_name_version", table_name="model_version_tags")
    op.drop_index("idx_model_versions_name_stage_version", table_name="model_versions")

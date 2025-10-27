"""increase_model_version_tag_value_limit

Create Date: 2025-06-23 11:05:41.676297

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "bda7b8c39065"
down_revision = "6953534de441"
branch_labels = None
depends_on = None


def upgrade():
    # Use batch mode for SQLite compatibility
    with op.batch_alter_table("model_version_tags") as batch_op:
        # Increase value column from VARCHAR(5000) to TEXT (unlimited)
        # We use Text type which maps appropriately for each database:
        # - PostgreSQL: TEXT (up to 1GB)
        # - MySQL: TEXT (up to 65,535 bytes) or LONGTEXT if needed
        # - SQLite: TEXT (no limit)
        # - MSSQL: VARCHAR(MAX) (up to 2GB)
        batch_op.alter_column(
            "value",
            existing_type=sa.String(5000),
            type_=sa.Text(),
            existing_nullable=True,
            existing_server_default=None,
        )


def downgrade():
    pass

"""increase_model_description_limit

Revision ID: 322b1ae00ba2
Revises: f5a4f2784254
Create Date: 2025-12-12 10:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "322b1ae00ba2"
down_revision = "f5a4f2784254"
branch_labels = None
depends_on = None


def upgrade():
    """
    Increase description column limit for registered_models and model_versions tables
    from VARCHAR(5000) to TEXT to support longer model descriptions.
    """
    # Use batch mode for SQLite compatibility
    with op.batch_alter_table("registered_models") as batch_op:
        # Increase description column from VARCHAR(5000) to TEXT (unlimited)
        # We use Text type which maps appropriately for each database:
        # - PostgreSQL: TEXT (up to 1GB)
        # - MySQL: TEXT (up to 65,535 bytes) or LONGTEXT if needed
        # - SQLite: TEXT (no limit)
        # - MSSQL: VARCHAR(MAX) (up to 2GB)
        batch_op.alter_column(
            "description",
            existing_type=sa.String(5000),
            type_=sa.Text(),
            existing_nullable=True,
            existing_server_default=None,
        )

    with op.batch_alter_table("model_versions") as batch_op:
        # Increase description column from VARCHAR(5000) to TEXT (unlimited)
        batch_op.alter_column(
            "description",
            existing_type=sa.String(5000),
            type_=sa.Text(),
            existing_nullable=True,
            existing_server_default=None,
        )


def downgrade():
    """
    Downgrade description columns back to VARCHAR(5000).
    Note: This may cause data truncation if descriptions exceed 5000 characters.
    """
    with op.batch_alter_table("registered_models") as batch_op:
        batch_op.alter_column(
            "description",
            existing_type=sa.Text(),
            type_=sa.String(5000),
            existing_nullable=True,
            existing_server_default=None,
        )

    with op.batch_alter_table("model_versions") as batch_op:
        batch_op.alter_column(
            "description",
            existing_type=sa.Text(),
            type_=sa.String(5000),
            existing_nullable=True,
            existing_server_default=None,
        )

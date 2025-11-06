"""add_secret_permissions

Revision ID: dfb437445764
Revises: 8606fa83a998
Create Date: 2025-01-04 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "dfb437445764"
down_revision = "8606fa83a998"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "secret_permissions",
        sa.Column("id", sa.Integer(), nullable=False, primary_key=True),
        sa.Column("secret_id", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("permission", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_secret_user_id"),
        sa.UniqueConstraint("secret_id", "user_id", name="unique_secret_user"),
    )


def downgrade() -> None:
    op.drop_table("secret_permissions")

"""add_scorer_permissions

Revision ID: 0965eb92f5f0
Revises: 8606fa83a998
Create Date: 2025-11-03 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0965eb92f5f0"
down_revision = "8606fa83a998"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "scorer_permissions",
        sa.Column("id", sa.Integer(), nullable=False, primary_key=True),
        sa.Column("experiment_id", sa.String(length=255), nullable=False),
        sa.Column("scorer_name", sa.String(length=256), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("permission", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_scorer_perm_user_id"),
        sa.UniqueConstraint("experiment_id", "scorer_name", "user_id", name="unique_scorer_user"),
    )


def downgrade() -> None:
    op.drop_table("scorer_permissions")

"""initial_migration

Revision ID: 8606fa83a998
Revises:
Create Date: 2023-07-07 23:30:50.921970

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "8606fa83a998"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False, primary_key=True),
        sa.Column("username", sa.String(length=255), nullable=True),
        sa.Column("password_hash", sa.String(length=255), nullable=True),
        sa.Column("is_admin", sa.Boolean(), nullable=True),
        sa.UniqueConstraint("username"),
    )
    op.create_table(
        "experiment_permissions",
        sa.Column("id", sa.Integer(), nullable=False, primary_key=True),
        sa.Column("experiment_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("permission", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_user_id"),
        sa.UniqueConstraint("experiment_id", "user_id", name="unique_experiment_user"),
    )
    op.create_table(
        "registered_model_permissions",
        sa.Column("id", sa.Integer(), nullable=False, primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("permission", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_user_id"),
        sa.UniqueConstraint("name", "user_id", name="unique_name_user"),
    )


def downgrade() -> None:
    op.drop_table("registered_model_permissions")
    op.drop_table("experiment_permissions")
    op.drop_table("users")

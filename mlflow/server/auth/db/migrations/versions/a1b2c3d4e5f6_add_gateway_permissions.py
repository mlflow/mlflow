"""add_gateway_permissions

Revision ID: a1b2c3d4e5f6
Revises: 0965eb92f5f0
Create Date: 2025-12-30 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "a1b2c3d4e5f6"
down_revision = "0965eb92f5f0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "gateway_secret_permissions",
        sa.Column("id", sa.Integer(), nullable=False, primary_key=True),
        sa.Column("secret_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("permission", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_gateway_secret_perm_user_id"),
        sa.UniqueConstraint("secret_id", "user_id", name="unique_secret_user"),
    )

    op.create_table(
        "gateway_endpoint_permissions",
        sa.Column("id", sa.Integer(), nullable=False, primary_key=True),
        sa.Column("endpoint_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("permission", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_gateway_endpoint_perm_user_id"),
        sa.UniqueConstraint("endpoint_id", "user_id", name="unique_endpoint_user"),
    )

    op.create_table(
        "gateway_model_definition_permissions",
        sa.Column("id", sa.Integer(), nullable=False, primary_key=True),
        sa.Column("model_definition_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("permission", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_gateway_model_def_perm_user_id"
        ),
        sa.UniqueConstraint("model_definition_id", "user_id", name="unique_model_def_user"),
    )


def downgrade() -> None:
    op.drop_table("gateway_model_definition_permissions")
    op.drop_table("gateway_endpoint_permissions")
    op.drop_table("gateway_secret_permissions")

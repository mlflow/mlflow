"""Add RBAC tables (roles, role_permissions, user_role_assignments)

Revision ID: c3d4e5f6a7b8
Revises: 2ed73881770d
Create Date: 2026-04-20 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "c3d4e5f6a7b8"
down_revision = "2ed73881770d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "roles",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("workspace", sa.String(length=63), nullable=False),
        sa.Column("description", sa.String(length=1024), nullable=True),
        sa.UniqueConstraint("workspace", "name", name="unique_workspace_role_name"),
    )
    op.create_index("idx_roles_workspace", "roles", ["workspace"], unique=False)

    op.create_table(
        "role_permissions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("role_id", sa.Integer(), sa.ForeignKey("roles.id"), nullable=False),
        sa.Column("resource_type", sa.String(length=64), nullable=False),
        sa.Column("resource_pattern", sa.String(length=255), nullable=False),
        sa.Column("permission", sa.String(length=255), nullable=False),
        sa.UniqueConstraint(
            "role_id", "resource_type", "resource_pattern", name="unique_role_resource_perm"
        ),
    )
    op.create_index("idx_role_permissions_role_id", "role_permissions", ["role_id"], unique=False)

    op.create_table(
        "user_role_assignments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("role_id", sa.Integer(), sa.ForeignKey("roles.id"), nullable=False),
        sa.UniqueConstraint("user_id", "role_id", name="unique_user_role"),
    )
    op.create_index(
        "idx_user_role_assignments_user_id", "user_role_assignments", ["user_id"], unique=False
    )
    op.create_index(
        "idx_user_role_assignments_role_id", "user_role_assignments", ["role_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("idx_user_role_assignments_role_id", table_name="user_role_assignments")
    op.drop_index("idx_user_role_assignments_user_id", table_name="user_role_assignments")
    op.drop_table("user_role_assignments")
    op.drop_index("idx_role_permissions_role_id", table_name="role_permissions")
    op.drop_table("role_permissions")
    op.drop_index("idx_roles_workspace", table_name="roles")
    op.drop_table("roles")

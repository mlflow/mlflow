"""Add workspace awareness to permissions

Revision ID: 2ed73881770d
Revises: a1b2c3d4e5f6
Create Date: 2026-01-16 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

# revision identifiers, used by Alembic.
revision = "2ed73881770d"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "workspace_permissions",
        sa.Column("workspace", sa.String(length=63), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("permission", sa.String(length=32), nullable=False),
        sa.PrimaryKeyConstraint("workspace", "user_id", name="workspace_permissions_pk"),
    )
    op.create_index(
        "idx_workspace_permissions_user_id", "workspace_permissions", ["user_id"], unique=False
    )
    op.create_index(
        "idx_workspace_permissions_workspace",
        "workspace_permissions",
        ["workspace"],
        unique=False,
    )

    with op.batch_alter_table("registered_model_permissions", recreate="auto") as batch_op:
        batch_op.add_column(
            sa.Column(
                "workspace",
                sa.String(length=63),
                nullable=False,
                server_default=sa.text(f"'{DEFAULT_WORKSPACE_NAME}'"),
            )
        )

    conn = op.get_bind()
    inspector = sa.inspect(conn)
    unique_constraints = inspector.get_unique_constraints("registered_model_permissions")
    has_unique_name_user = any(
        constraint.get("name") == "unique_name_user" for constraint in unique_constraints
    )

    with op.batch_alter_table("registered_model_permissions", recreate="auto") as batch_op:
        if has_unique_name_user:
            batch_op.drop_constraint("unique_name_user", type_="unique")
        batch_op.create_unique_constraint(
            "unique_workspace_name_user", ["workspace", "name", "user_id"]
        )


def downgrade() -> None:
    conn = op.get_bind()
    conflicts = conn.execute(
        sa.text(
            """
            SELECT name, user_id, COUNT(DISTINCT workspace) AS workspace_count
            FROM registered_model_permissions
            GROUP BY name, user_id
            HAVING COUNT(DISTINCT workspace) > 1
            """
        )
    ).fetchall()
    if conflicts:
        details = ", ".join(f"(name={row.name}, user_id={row.user_id})" for row in conflicts)
        raise RuntimeError(
            "Cannot downgrade workspace permissions migration because dropping the workspace "
            + "column would create conflicts for the following registered_model_permissions rows: "
            f"{details}. Please merge or delete the conflicting rows before retrying the downgrade."
        )

    op.drop_index("idx_workspace_permissions_workspace", table_name="workspace_permissions")
    op.drop_index("idx_workspace_permissions_user_id", table_name="workspace_permissions")
    op.drop_table("workspace_permissions")
    with op.batch_alter_table("registered_model_permissions", recreate="auto") as batch_op:
        batch_op.drop_constraint("unique_workspace_name_user", type_="unique")
        batch_op.create_unique_constraint("unique_name_user", ["name", "user_id"])
        batch_op.drop_column("workspace")

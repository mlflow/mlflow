"""Add ON DELETE CASCADE to all FKs referencing users.id

Revision ID: e6c5d2f7a8b9
Revises: c3d4e5f6a7b8
Create Date: 2026-04-27 09:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "e6c5d2f7a8b9"
down_revision = "c3d4e5f6a7b8"
branch_labels = None
depends_on = None


# Each entry is (table_name, existing_fk_name).
#
# Tables whose `user_id` FK references `users.id`. Names taken from the
# migrations that originally created each table. The two unnamed FKs
# (workspace_permissions, user_role_assignments) are addressed via the
# `naming_convention` synthesized name below.
_NAMED_FKS = (
    ("experiment_permissions", "fk_user_id"),
    ("registered_model_permissions", "fk_user_id"),
    ("scorer_permissions", "fk_scorer_perm_user_id"),
    ("gateway_secret_permissions", "fk_gateway_secret_perm_user_id"),
    ("gateway_endpoint_permissions", "fk_gateway_endpoint_perm_user_id"),
    ("gateway_model_definition_permissions", "fk_gateway_model_def_perm_user_id"),
)
_UNNAMED_FK_TABLES = (
    "workspace_permissions",
    "user_role_assignments",
)


_NAMING_CONVENTION = {
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
}


def _new_fk_name(table_name: str) -> str:
    return f"fk_{table_name}_user_id_users"


def _alter_named_fk(table_name: str, old_fk_name: str, ondelete: str | None) -> None:
    new_fk_name = _new_fk_name(table_name)
    dialect = op.get_context().dialect.name
    if dialect == "sqlite":
        with op.batch_alter_table(table_name) as batch_op:
            batch_op.drop_constraint(old_fk_name, type_="foreignkey")
            batch_op.create_foreign_key(
                new_fk_name, "users", ["user_id"], ["id"], ondelete=ondelete
            )
    else:
        op.drop_constraint(old_fk_name, table_name, type_="foreignkey")
        op.create_foreign_key(
            new_fk_name, table_name, "users", ["user_id"], ["id"], ondelete=ondelete
        )


def _alter_unnamed_fk(table_name: str, ondelete: str | None) -> None:
    new_fk_name = _new_fk_name(table_name)
    dialect = op.get_context().dialect.name
    if dialect == "sqlite":
        # `naming_convention` causes Alembic to address the existing unnamed
        # FK by the synthesized name (which equals `new_fk_name`).
        with op.batch_alter_table(table_name, naming_convention=_NAMING_CONVENTION) as batch_op:
            batch_op.drop_constraint(new_fk_name, type_="foreignkey")
            batch_op.create_foreign_key(
                new_fk_name, "users", ["user_id"], ["id"], ondelete=ondelete
            )
    else:
        # Other dialects: introspect and drop by actual name.
        conn = op.get_bind()
        inspector = sa.inspect(conn)
        existing_fk = next(
            (
                fk
                for fk in inspector.get_foreign_keys(table_name)
                if fk["referred_table"] == "users" and fk["constrained_columns"] == ["user_id"]
            ),
            None,
        )
        if existing_fk and existing_fk.get("name"):
            op.drop_constraint(existing_fk["name"], table_name, type_="foreignkey")
        op.create_foreign_key(
            new_fk_name, table_name, "users", ["user_id"], ["id"], ondelete=ondelete
        )


def upgrade() -> None:
    for table_name, old_fk_name in _NAMED_FKS:
        _alter_named_fk(table_name, old_fk_name, ondelete="CASCADE")
    for table_name in _UNNAMED_FK_TABLES:
        _alter_unnamed_fk(table_name, ondelete="CASCADE")


def downgrade() -> None:
    # Restore the original (no ON DELETE) FKs. The downgrade names them
    # with the convention so subsequent upgrades can find them.
    for table_name, old_fk_name in _NAMED_FKS:
        # Drop the cascading FK we created in upgrade(), then restore the
        # original-named FK without ondelete.
        new_fk_name = _new_fk_name(table_name)
        dialect = op.get_context().dialect.name
        if dialect == "sqlite":
            with op.batch_alter_table(table_name) as batch_op:
                batch_op.drop_constraint(new_fk_name, type_="foreignkey")
                batch_op.create_foreign_key(old_fk_name, "users", ["user_id"], ["id"])
        else:
            op.drop_constraint(new_fk_name, table_name, type_="foreignkey")
            op.create_foreign_key(old_fk_name, table_name, "users", ["user_id"], ["id"])
    for table_name in _UNNAMED_FK_TABLES:
        _alter_unnamed_fk(table_name, ondelete=None)

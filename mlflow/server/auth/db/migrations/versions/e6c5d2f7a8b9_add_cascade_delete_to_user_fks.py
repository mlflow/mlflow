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


# Tables whose `user_id` FK references `users.id`. We can't hardcode the
# existing FK names because the same logical table can have one of three
# shapes in the wild:
#   * named FK from the per-table create migration (e.g. `fk_user_id`,
#     `fk_scorer_perm_user_id`)
#   * unnamed FK from a migration that used `sa.ForeignKey("users.id")`
#     (workspace_permissions, user_role_assignments)
#   * unnamed FK from a legacy pre-Alembic schema that this codebase
#     stamps to the initial revision (`experiment_permissions`,
#     `registered_model_permissions` — see `test_upgrade_from_legacy_database`)
# The migration introspects the actual FK at upgrade time and drops it by
# name, falling back to a synthesized name (under `naming_convention`) for
# unnamed FKs.
_TABLES = (
    "experiment_permissions",
    "registered_model_permissions",
    "scorer_permissions",
    "gateway_secret_permissions",
    "gateway_endpoint_permissions",
    "gateway_model_definition_permissions",
    "workspace_permissions",
    "user_role_assignments",
)


_NAMING_CONVENTION = {
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
}


def _new_fk_name(table_name: str) -> str:
    return f"fk_{table_name}_user_id_users"


def _existing_user_fk_name(table_name: str) -> str | None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    for fk in inspector.get_foreign_keys(table_name):
        if fk["referred_table"] == "users" and fk["constrained_columns"] == ["user_id"]:
            return fk.get("name")
    return None


def _alter_user_fk(table_name: str, ondelete: str | None) -> None:
    new_fk_name = _new_fk_name(table_name)
    existing_name = _existing_user_fk_name(table_name)
    drop_name = existing_name or new_fk_name
    dialect = op.get_context().dialect.name
    if dialect == "sqlite":
        # `naming_convention` lets `batch_alter_table` address an unnamed FK
        # by the synthesized name. Named FKs keep their original name and
        # are dropped by that.
        with op.batch_alter_table(table_name, naming_convention=_NAMING_CONVENTION) as batch_op:
            batch_op.drop_constraint(drop_name, type_="foreignkey")
            batch_op.create_foreign_key(
                new_fk_name, "users", ["user_id"], ["id"], ondelete=ondelete
            )
    else:
        # Other dialects can ALTER directly. Skip the drop if the FK is
        # unnamed and inspection didn't yield one (Postgres/MySQL always
        # name FKs, so this branch should not be hit in practice).
        if existing_name:
            op.drop_constraint(existing_name, table_name, type_="foreignkey")
        op.create_foreign_key(
            new_fk_name, table_name, "users", ["user_id"], ["id"], ondelete=ondelete
        )


def upgrade() -> None:
    for table_name in _TABLES:
        _alter_user_fk(table_name, ondelete="CASCADE")


def downgrade() -> None:
    # The original FK had no ON DELETE clause; recreate it that way. The
    # downgrade re-uses the same `_alter_user_fk` helper, since the
    # introspection logic identifies the upgraded FK (by `_new_fk_name`)
    # equally well.
    for table_name in _TABLES:
        _alter_user_fk(table_name, ondelete=None)

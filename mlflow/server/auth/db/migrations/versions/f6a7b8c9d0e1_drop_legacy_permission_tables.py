"""Drop pre-RBAC permission tables

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-05-06 17:00:00.000000

The graduation migration for the RBAC simplification stack. ``e5f6a7b8c9d0``
backfilled every legacy per-resource permission row into ``role_permissions``
and stopped the auth server from reading or writing the legacy tables, but
**retained** the seven pre-RBAC tables on disk so operators could roll back
without restoring from backup. This migration removes them now that the
simplified model has bedded in.

The seven tables dropped here, with their introducing migration:

- ``experiment_permissions``           — ``8606fa83a998_initial_migration``
- ``registered_model_permissions``     — ``8606fa83a998_initial_migration``
- ``scorer_permissions``               — ``0965eb92f5f0_add_scorer_permissions``
- ``gateway_secret_permissions``       — ``a1b2c3d4e5f6_add_gateway_permissions``
- ``gateway_endpoint_permissions``     — ``a1b2c3d4e5f6_add_gateway_permissions``
- ``gateway_model_definition_permissions`` — ``a1b2c3d4e5f6_add_gateway_permissions``
- ``workspace_permissions``            — ``2ed73881770d_workspace_permissions``

After this migration:

- ``SqlAlchemyStore.delete_user`` no longer needs to scrub legacy rows; the
  cascade through ``SqlUser.user_role_assignments`` is sufficient.
- The ``_RETAINED_LEGACY_PERMISSION_TABLES`` constant in ``sqlalchemy_store.py``
  can become ``()`` (or be removed); the corresponding test
  ``test_delete_user_clears_retained_legacy_permission_rows`` is no longer
  reachable and should be dropped.

Downgrade rebuilds the table schemas (so subsequent ``alembic downgrade`` calls
that rebuild prior revisions can stamp the schema correctly), but does **not**
restore the row data — that's the point-of-no-return contract this migration
ships under. Operators rolling back to a state that needs the legacy data must
restore from backup.

Tracking: https://github.com/mlflow/mlflow/issues/23087
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "f6a7b8c9d0e1"
down_revision = "e5f6a7b8c9d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Order matters under non-cascading FKs: child tables before any parent.
    # All seven tables FK only to ``users.id`` (which we leave alone), so the
    # internal order between them is purely cosmetic — keep it alphabetic.
    op.drop_table("experiment_permissions")
    op.drop_table("gateway_endpoint_permissions")
    op.drop_table("gateway_model_definition_permissions")
    op.drop_table("gateway_secret_permissions")
    op.drop_table("registered_model_permissions")
    op.drop_table("scorer_permissions")
    op.drop_table("workspace_permissions")


def downgrade() -> None:
    # Rebuild the table schemas so older migrations referencing them remain
    # introspectable. The shapes match each table's introducing migration
    # exactly. Row data is *not* restored — see module docstring.
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
        sa.Column("workspace", sa.String(length=255), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("permission", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_user_id"),
        sa.UniqueConstraint("workspace", "name", "user_id", name="unique_name_user"),
    )
    op.create_table(
        "scorer_permissions",
        sa.Column("id", sa.Integer(), nullable=False, primary_key=True),
        sa.Column("experiment_id", sa.String(length=255), nullable=False),
        sa.Column("scorer_name", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("permission", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_user_id"),
        sa.UniqueConstraint("experiment_id", "scorer_name", "user_id", name="unique_scorer_user"),
    )
    op.create_table(
        "gateway_secret_permissions",
        sa.Column("id", sa.Integer(), nullable=False, primary_key=True),
        sa.Column("secret_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("permission", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_user_id"),
        sa.UniqueConstraint("secret_id", "user_id", name="unique_secret_user"),
    )
    op.create_table(
        "gateway_endpoint_permissions",
        sa.Column("id", sa.Integer(), nullable=False, primary_key=True),
        sa.Column("endpoint_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("permission", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_user_id"),
        sa.UniqueConstraint("endpoint_id", "user_id", name="unique_endpoint_user"),
    )
    op.create_table(
        "gateway_model_definition_permissions",
        sa.Column("id", sa.Integer(), nullable=False, primary_key=True),
        sa.Column("model_definition_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("permission", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_user_id"),
        sa.UniqueConstraint("model_definition_id", "user_id", name="unique_model_definition_user"),
    )
    op.create_table(
        "workspace_permissions",
        sa.Column("id", sa.Integer(), nullable=False, primary_key=True),
        sa.Column("workspace", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("permission", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_user_id"),
        sa.UniqueConstraint("workspace", "user_id", name="unique_workspace_user"),
    )

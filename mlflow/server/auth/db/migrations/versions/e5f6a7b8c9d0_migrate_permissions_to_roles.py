"""Migrate legacy per-resource permissions into role_permissions

Revision ID: e5f6a7b8c9d0
Revises: c3d4e5f6a7b8
Create Date: 2026-04-24 00:00:00.000000

Collapses the pre-RBAC per-resource permission tables
(``experiment_permissions``, ``registered_model_permissions``, ``scorer_permissions``,
``gateway_secret_permissions``, ``gateway_endpoint_permissions``,
``gateway_model_definition_permissions``, ``workspace_permissions``) into
``role_permissions`` rows hung off a synthetic ``__user_<id>__`` role per
``(user, workspace)`` pair, then drops the legacy tables.

After this migration, ``role_permissions`` is the sole source of truth for user
permissions; the auth server reads and writes only that table.

Workspace scoping for legacy tables without a ``workspace`` column
(``experiment_permissions``, ``scorer_permissions``, ``gateway_*``): those
grants were workspace-agnostic at the table level, so they land in the
``DEFAULT_WORKSPACE_NAME`` synthetic role.

"""

import re
from urllib.parse import quote

import sqlalchemy as sa
from alembic import op
from sqlalchemy import bindparam, text

from mlflow.exceptions import MlflowException
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

# revision identifiers, used by Alembic.
revision = "e5f6a7b8c9d0"
down_revision = "c3d4e5f6a7b8"
branch_labels = None
depends_on = None


_SYNTHETIC_ROLE_NAME = "__user_{user_id}__"
# Exact match for the reserved synthetic role namespace. Used both to detect
# pre-existing collisions during upgrade (see ``_get_or_create_synthetic_role``)
# and to scope the downgrade cleanup so it never removes an admin-created role
# that merely happens to start/end with underscores.
_SYNTHETIC_ROLE_NAME_RE = re.compile(r"^__user_\d+__$")


def _get_or_create_synthetic_role(conn, user_id: int, workspace: str) -> int:
    name = _SYNTHETIC_ROLE_NAME.format(user_id=user_id)
    row = conn.execute(
        text("SELECT id FROM roles WHERE workspace = :workspace AND name = :name"),
        {"workspace": workspace, "name": name},
    ).first()
    if row is not None:
        role_id = row[0]
        # A role with this exact reserved name already exists. Make sure it is
        # not a user-created role that happens to collide with the synthetic
        # namespace and has other users assigned — attaching this user's
        # migrated grants to such a role (and assigning them to it) would leak
        # grants across users. If the only assignment is for ``user_id`` (or no
        # assignments yet), treat it as safe to reuse.
        other_assignees = conn.execute(
            text(
                "SELECT user_id FROM user_role_assignments "
                "WHERE role_id = :role_id AND user_id != :user_id"
            ),
            {"role_id": role_id, "user_id": user_id},
        ).first()
        if other_assignees is not None:
            raise MlflowException.invalid_parameter_value(
                f"Role {name!r} in workspace {workspace!r} collides with the reserved "
                f"synthetic namespace '__user_<id>__' but is assigned to other users "
                f"(e.g. user_id={other_assignees[0]}). Rename the conflicting role "
                "before running this migration."
            )
    else:
        conn.execute(
            text(
                "INSERT INTO roles (name, workspace, description) VALUES (:name, :workspace, NULL)"
            ),
            {"name": name, "workspace": workspace},
        )
        role_id = conn.execute(
            text("SELECT id FROM roles WHERE workspace = :workspace AND name = :name"),
            {"workspace": workspace, "name": name},
        ).scalar_one()
    assignment = conn.execute(
        text(
            "SELECT id FROM user_role_assignments WHERE user_id = :user_id AND role_id = :role_id"
        ),
        {"user_id": user_id, "role_id": role_id},
    ).first()
    if assignment is None:
        conn.execute(
            text(
                "INSERT INTO user_role_assignments (user_id, role_id) VALUES (:user_id, :role_id)"
            ),
            {"user_id": user_id, "role_id": role_id},
        )
    return role_id


def _insert_role_permission_if_missing(
    conn, role_id: int, resource_type: str, resource_pattern: str, permission: str
) -> None:
    existing = conn.execute(
        text(
            "SELECT id FROM role_permissions "
            "WHERE role_id = :role_id "
            "AND resource_type = :resource_type "
            "AND resource_pattern = :resource_pattern"
        ),
        {
            "role_id": role_id,
            "resource_type": resource_type,
            "resource_pattern": resource_pattern,
        },
    ).first()
    if existing is not None:
        return
    conn.execute(
        text(
            "INSERT INTO role_permissions "
            "(role_id, resource_type, resource_pattern, permission) "
            "VALUES (:role_id, :resource_type, :resource_pattern, :permission)"
        ),
        {
            "role_id": role_id,
            "resource_type": resource_type,
            "resource_pattern": resource_pattern,
            "permission": permission,
        },
    )


def _backfill_table(conn, select_sql: str, row_to_mirror) -> None:
    for row in conn.execute(text(select_sql)):
        user_id, workspace, resource_type, resource_pattern, permission = row_to_mirror(row)
        role_id = _get_or_create_synthetic_role(conn, user_id, workspace)
        _insert_role_permission_if_missing(
            conn, role_id, resource_type, resource_pattern, permission
        )


def upgrade() -> None:
    conn = op.get_bind()
    default = DEFAULT_WORKSPACE_NAME

    # workspace_permissions → (resource_type='*', resource_pattern='*', permission)
    _backfill_table(
        conn,
        "SELECT user_id, workspace, permission FROM workspace_permissions",
        lambda r: (r.user_id, r.workspace, "*", "*", r.permission),
    )
    # experiment_permissions → (experiment, <experiment_id>, permission) in default ws
    _backfill_table(
        conn,
        "SELECT user_id, experiment_id, permission FROM experiment_permissions",
        lambda r: (r.user_id, default, "experiment", r.experiment_id, r.permission),
    )
    # registered_model_permissions → (registered_model, <name>, permission) in row's ws
    _backfill_table(
        conn,
        "SELECT user_id, workspace, name, permission FROM registered_model_permissions",
        lambda r: (r.user_id, r.workspace, "registered_model", r.name, r.permission),
    )
    # scorer_permissions → (scorer, <exp_id>/<quote(scorer_name)>, permission) in default ws.
    # Scorer names may contain ``/`` (see ``validate_scorer_name`` which only rejects
    # empty/whitespace), so the name component is URL-encoded to keep the compound
    # key unambiguous. Must match ``SqlAlchemyStore._scorer_pattern`` exactly or
    # post-migration lookups won't find these rows.
    _backfill_table(
        conn,
        "SELECT user_id, experiment_id, scorer_name, permission FROM scorer_permissions",
        lambda r: (
            r.user_id,
            default,
            "scorer",
            f"{r.experiment_id}/{quote(r.scorer_name, safe='')}",
            r.permission,
        ),
    )
    # gateway_secret_permissions → (gateway_secret, <secret_id>, permission)
    _backfill_table(
        conn,
        "SELECT user_id, secret_id, permission FROM gateway_secret_permissions",
        lambda r: (r.user_id, default, "gateway_secret", r.secret_id, r.permission),
    )
    # gateway_endpoint_permissions → (gateway_endpoint, <endpoint_id>, permission)
    _backfill_table(
        conn,
        "SELECT user_id, endpoint_id, permission FROM gateway_endpoint_permissions",
        lambda r: (r.user_id, default, "gateway_endpoint", r.endpoint_id, r.permission),
    )
    # gateway_model_definition_permissions → (gateway_model_definition, <id>, permission)
    _backfill_table(
        conn,
        (
            "SELECT user_id, model_definition_id, permission "
            "FROM gateway_model_definition_permissions"
        ),
        lambda r: (
            r.user_id,
            default,
            "gateway_model_definition",
            r.model_definition_id,
            r.permission,
        ),
    )

    # Drop the legacy per-resource permission tables now that all data has been
    # copied into ``role_permissions`` under synthetic ``__user_<id>__`` roles and the
    # store/handler code no longer reads them.
    op.drop_table("experiment_permissions")
    op.drop_table("registered_model_permissions")
    op.drop_table("scorer_permissions")
    op.drop_table("gateway_secret_permissions")
    op.drop_table("gateway_endpoint_permissions")
    op.drop_table("gateway_model_definition_permissions")
    op.drop_table("workspace_permissions")


def downgrade() -> None:
    # Re-create the legacy table schemas so an operator could restore pre-migration
    # data from backup, then remove the synthetic ``__user_<id>__`` roles this
    # migration created. Data is NOT re-populated from role_permissions — operators
    # must restore from a DB backup taken before the upgrade.
    from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME as _DEFAULT_WS

    op.create_table(
        "experiment_permissions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("experiment_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("permission", sa.String(length=255)),
        sa.UniqueConstraint("experiment_id", "user_id", name="unique_experiment_user"),
    )
    op.create_table(
        "registered_model_permissions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "workspace",
            sa.String(length=63),
            nullable=False,
            server_default=sa.text(f"'{_DEFAULT_WS}'"),
        ),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("permission", sa.String(length=255)),
        sa.UniqueConstraint("workspace", "name", "user_id", name="unique_workspace_name_user"),
    )
    op.create_table(
        "scorer_permissions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("experiment_id", sa.String(length=255), nullable=False),
        sa.Column("scorer_name", sa.String(length=256), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("permission", sa.String(length=255)),
        sa.UniqueConstraint("experiment_id", "scorer_name", "user_id", name="unique_scorer_user"),
    )
    op.create_table(
        "gateway_secret_permissions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("secret_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("permission", sa.String(length=255)),
        sa.UniqueConstraint("secret_id", "user_id", name="unique_secret_user"),
    )
    op.create_table(
        "gateway_endpoint_permissions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("endpoint_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("permission", sa.String(length=255)),
        sa.UniqueConstraint("endpoint_id", "user_id", name="unique_endpoint_user"),
    )
    op.create_table(
        "gateway_model_definition_permissions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_definition_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("permission", sa.String(length=255)),
        sa.UniqueConstraint("model_definition_id", "user_id", name="unique_model_def_user"),
    )
    op.create_table(
        "workspace_permissions",
        sa.Column("workspace", sa.String(length=63), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("permission", sa.String(length=32), nullable=False),
        sa.PrimaryKeyConstraint("workspace", "user_id", name="workspace_permissions_pk"),
        sa.Index("idx_workspace_permissions_user_id", "user_id"),
        sa.Index("idx_workspace_permissions_workspace", "workspace"),
    )

    # Remove all synthetic per-user roles this migration could have created. Filter
    # by the exact ``^__user_\d+__$`` pattern so we don't accidentally nuke admin-
    # created roles that merely start/end with underscores (e.g. ``__user_admin__``).
    # Portable via regex in Python (LIKE with ESCAPE varies across dialects).
    conn = op.get_bind()
    role_ids = [
        row_id
        for (row_id, name) in conn.execute(text("SELECT id, name FROM roles"))
        if _SYNTHETIC_ROLE_NAME_RE.match(name)
    ]
    if not role_ids:
        return
    conn.execute(
        text("DELETE FROM role_permissions WHERE role_id IN :ids").bindparams(
            bindparam("ids", expanding=True)
        ),
        {"ids": role_ids},
    )
    conn.execute(
        text("DELETE FROM user_role_assignments WHERE role_id IN :ids").bindparams(
            bindparam("ids", expanding=True)
        ),
        {"ids": role_ids},
    )
    conn.execute(
        text("DELETE FROM roles WHERE id IN :ids").bindparams(bindparam("ids", expanding=True)),
        {"ids": role_ids},
    )

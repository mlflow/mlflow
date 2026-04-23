"""Backfill role_permissions from legacy per-resource permission tables

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-04-23 00:00:00.000000

For each user with grants in the legacy per-resource permission tables, create a
synthetic ``__user_<id>__`` role in the appropriate workspace and mirror every grant
into ``role_permissions``. Paired with the M1 / M1.5 dual-write, this lets later
phases flip reads onto ``role_permissions`` as the sole source of truth.

The migration is idempotent: each mirrored row is inserted via SELECT-then-INSERT
so re-runs are safe.

Workspace scoping:
- ``workspace_permissions``: the workspace is in the row itself.
- ``registered_model_permissions``: the workspace is in the row itself.
- ``experiment_permissions``, ``scorer_permissions``, ``gateway_*_permissions``:
  these tables do not carry a workspace column. The legacy reader also ignores
  workspace for these tables, so we mirror them into ``DEFAULT_WORKSPACE_NAME``.
  In multi-workspace deployments this places pre-M1 grants in the default
  workspace's synthetic role; post-M1 dual-write already captures the grant's
  workspace at request time.

"""

from alembic import op
from sqlalchemy import bindparam, text

from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

# revision identifiers, used by Alembic.
revision = "d4e5f6a7b8c9"
down_revision = "c3d4e5f6a7b8"
branch_labels = None
depends_on = None


_SYNTHETIC_ROLE_NAME = "__user_{user_id}__"


def _get_or_create_synthetic_role(conn, user_id: int, workspace: str) -> int:
    name = _SYNTHETIC_ROLE_NAME.format(user_id=user_id)
    row = conn.execute(
        text("SELECT id FROM roles WHERE workspace = :workspace AND name = :name"),
        {"workspace": workspace, "name": name},
    ).first()
    if row is not None:
        role_id = row[0]
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
    # Ensure the user is assigned.
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


def _backfill_table(
    conn,
    select_sql: str,
    row_to_mirror,
) -> None:
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

    # scorer_permissions → (scorer, <exp_id>/<scorer_name>, permission) in default ws
    _backfill_table(
        conn,
        "SELECT user_id, experiment_id, scorer_name, permission FROM scorer_permissions",
        lambda r: (
            r.user_id,
            default,
            "scorer",
            f"{r.experiment_id}/{r.scorer_name}",
            r.permission,
        ),
    )

    # gateway_secret_permissions → (gateway_secret, <secret_id>, permission) in default ws
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


def downgrade() -> None:
    # Remove all synthetic per-user roles this migration could have created. The FK from
    # role_permissions / user_role_assignments to roles doesn't declare ON DELETE CASCADE
    # at the DB level, so delete child rows explicitly. Filter by the synthetic name
    # pattern in Python (portable across dialects — cross-dialect LIKE with underscores
    # and ESCAPE is error-prone).
    conn = op.get_bind()
    role_ids = [
        row_id
        for (row_id, name) in conn.execute(text("SELECT id, name FROM roles"))
        if name.startswith("__user_") and name.endswith("__")
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

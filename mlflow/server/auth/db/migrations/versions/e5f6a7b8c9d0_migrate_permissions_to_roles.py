"""Migrate legacy per-resource permissions into role_permissions

Revision ID: e5f6a7b8c9d0
Revises: c3d4e5f6a7b8
Create Date: 2026-04-24 00:00:00.000000

Backfills the pre-RBAC per-resource permission tables
(``experiment_permissions``, ``registered_model_permissions``, ``scorer_permissions``,
``gateway_secret_permissions``, ``gateway_endpoint_permissions``,
``gateway_model_definition_permissions``, ``workspace_permissions``) into
``role_permissions`` rows hung off a synthetic ``__user_<id>__`` role per
``(user, workspace)`` pair. The legacy tables are intentionally **not** dropped —
they remain in place as a paused snapshot so operators can roll back manually
without restoring from backup. A follow-up migration
(``f6a7b8c9d0e1_drop_legacy_permission_tables``, scheduled for MLflow 3.X+2 —
tracking: https://github.com/mlflow/mlflow/issues/23087) will retire them once
the simplified RBAC model has bedded in. Until then, ``SqlAlchemyStore.delete_user``
scrubs each user's rows from these tables on delete to satisfy the non-cascading
FKs from earlier migrations — see ``_RETAINED_LEGACY_PERMISSION_TABLES`` in
``sqlalchemy_store.py`` for the runtime side of this contract.

After this migration, ``role_permissions`` is the sole source of truth that the
auth server reads and writes; the legacy tables are no longer consulted.

The simplified permission model is enforced during backfill:

- Resource-level rows (``experiment``, ``registered_model``, ``scorer``,
  ``gateway_*``) with ``permission='NO_PERMISSIONS'`` are skipped — an absent
  grant combined with the configured ``default_permission`` already expresses
  "no access", so an explicit deny row is no longer supported.
- Workspace-level rows are normalised to the two-tier USE / MANAGE model:
  ``NO_PERMISSIONS`` rows are skipped; ``READ`` rows rewrite to a single
  ``USE`` grant (the pre-simplification "see workspace but cannot create" tier
  is no longer expressible as a single grant); ``EDIT`` rows fan out to one
  workspace-wide ``USE`` grant plus a type-wildcard ``EDIT`` grant on every
  concrete resource type, preserving the user's effective per-resource EDIT
  capability without smuggling EDIT into the workspace tier itself.

Workspace scoping for legacy tables without a ``workspace`` column
(``experiment_permissions``, ``scorer_permissions``, ``gateway_*``): those
grants were workspace-agnostic at the table level, so they land in the
``DEFAULT_WORKSPACE_NAME`` synthetic role.

"""

import re
from urllib.parse import quote

from alembic import op
from sqlalchemy import bindparam, text

from mlflow.exceptions import MlflowException
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

# Resource-level rows carrying this permission are dropped during backfill.
# See module docstring for rationale.
_DROPPED_PERMISSION = "NO_PERMISSIONS"

# Mapping applied to workspace-level rows during backfill to enforce the simplified
# USE/MANAGE-only workspace tier. ``READ`` rewrites to ``USE`` because the legacy
# "see workspace but cannot create" tier is no longer expressible at workspace scope.
# ``EDIT`` is handled specially in ``_workspace_row`` (fan-out to type-wildcard EDIT
# grants) so it does not appear here. Rows whose value is not in this map (i.e. USE,
# MANAGE) are migrated unchanged; ``NO_PERMISSIONS`` is skipped.
_LEGACY_WORKSPACE_PERMISSION_REWRITE = {
    "READ": "USE",
}

# Concrete resource types that ``workspace_permissions(EDIT)`` fans out to.
# Pre-simplification, a workspace EDIT grant gave the user EDIT on every resource
# in the workspace via the workspace-level fallback. The simplified model has no
# workspace-scope EDIT (workspace tier is USE/MANAGE only), so the migration
# preserves the capability by emitting one type-wildcard EDIT grant per resource
# type plus a workspace-wide USE grant for create rights and visibility.
_FAN_OUT_RESOURCE_TYPES = (
    "experiment",
    "registered_model",
    "scorer",
    "gateway_secret",
    "gateway_endpoint",
    "gateway_model_definition",
)

# Permission values legal in ``role_permissions`` post-migration. Hardcoded here
# (rather than imported from ``mlflow.server.auth.permissions``) because Alembic
# migrations are frozen snapshots — they must keep working even if the application
# code's permission set later drifts.
_VALID_RESOURCE_PERMISSIONS = frozenset({"READ", "USE", "EDIT", "MANAGE"})
_VALID_WORKSPACE_PERMISSIONS = frozenset({"USE", "MANAGE"})

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
    """Copy rows from a legacy table into ``role_permissions``.

    ``row_to_mirror`` returns a (possibly empty) list of grant tuples for each
    legacy row. Empty list = drop. Multiple tuples = fan out (used for
    workspace ``EDIT`` rows; see ``_workspace_row``).
    """
    for row in conn.execute(text(select_sql)):
        for mirrored in row_to_mirror(row):
            user_id, workspace, resource_type, resource_pattern, permission = mirrored
            role_id = _get_or_create_synthetic_role(conn, user_id, workspace)
            _insert_role_permission_if_missing(
                conn, role_id, resource_type, resource_pattern, permission
            )


def _resource_row(row, workspace, resource_type, resource_pattern):
    """Mirror a concrete-resource legacy row, dropping ``NO_PERMISSIONS`` denies.

    Rows whose ``permission`` is not in the simplified resource grant set
    (``READ``/``USE``/``EDIT``/``MANAGE``) are dropped — defensive against
    corrupted legacy data. ``NO_PERMISSIONS`` is the only such value we
    expect to see (it's documented as the deny form), but anything else
    would also be invalid post-migration. Returns a list with at most one
    tuple (empty = drop) so the call site can handle resource and workspace
    rows uniformly.
    """
    if row.permission not in _VALID_RESOURCE_PERMISSIONS:
        return []
    return [(row.user_id, workspace, resource_type, resource_pattern, row.permission)]


def _workspace_row(row):
    """Mirror a ``workspace_permissions`` row into one or more role_permissions grants.

    The simplified workspace tier is USE / MANAGE only, so legacy ``READ`` and
    ``EDIT`` don't have a single equivalent shape:

    - ``NO_PERMISSIONS`` is skipped.
    - ``READ`` is rewritten to a single ``('workspace', '*', USE)`` grant. The
      pre-simplification "see workspace but cannot create" semantic is no
      longer expressible as a single grant; operators that relied on it should
      switch to ``default_permission=READ`` after migration.
    - ``EDIT`` fans out to one workspace-wide ``('workspace', '*', USE)`` grant
      plus a type-wildcard ``EDIT`` grant on every concrete resource type, so
      the user keeps the same effective per-resource capability they had
      pre-simplification. The workspace ``USE`` row is what gives them
      workspace visibility and create rights; the type-wildcard ``EDIT`` rows
      are what the resource resolver matches on for read/update/etc.
    - ``USE`` and ``MANAGE`` migrate as a single ``('workspace', '*', PERMISSION)``
      grant unchanged.

    Returns a list of ``(user_id, workspace, resource_type, resource_pattern,
    permission)`` tuples (possibly empty for dropped rows).
    """
    if row.permission == _DROPPED_PERMISSION:
        return []
    if row.permission == "EDIT":
        return [
            (row.user_id, row.workspace, "workspace", "*", "USE"),
            *((row.user_id, row.workspace, rt, "*", "EDIT") for rt in _FAN_OUT_RESOURCE_TYPES),
        ]
    permission = _LEGACY_WORKSPACE_PERMISSION_REWRITE.get(row.permission, row.permission)
    if permission not in _VALID_WORKSPACE_PERMISSIONS:
        return []
    return [(row.user_id, row.workspace, "workspace", "*", permission)]


def upgrade() -> None:
    conn = op.get_bind()
    default = DEFAULT_WORKSPACE_NAME

    # workspace_permissions → (resource_type='workspace', resource_pattern='*', permission)
    # plus per-type EDIT fan-out for legacy EDIT rows; see ``_workspace_row``.
    _backfill_table(
        conn,
        "SELECT user_id, workspace, permission FROM workspace_permissions",
        _workspace_row,
    )
    # experiment_permissions → (experiment, <experiment_id>, permission) in default ws
    _backfill_table(
        conn,
        "SELECT user_id, experiment_id, permission FROM experiment_permissions",
        lambda r: _resource_row(r, default, "experiment", r.experiment_id),
    )
    # registered_model_permissions → (registered_model, <name>, permission) in row's ws
    _backfill_table(
        conn,
        "SELECT user_id, workspace, name, permission FROM registered_model_permissions",
        lambda r: _resource_row(r, r.workspace, "registered_model", r.name),
    )
    # scorer_permissions → (scorer, <exp_id>/<quote(scorer_name)>, permission) in default ws.
    # Scorer names may contain ``/`` (see ``validate_scorer_name`` which only rejects
    # empty/whitespace), so the name component is URL-encoded to keep the compound
    # key unambiguous. Must match ``SqlAlchemyStore._scorer_pattern`` exactly or
    # post-migration lookups won't find these rows.
    _backfill_table(
        conn,
        "SELECT user_id, experiment_id, scorer_name, permission FROM scorer_permissions",
        lambda r: _resource_row(
            r,
            default,
            "scorer",
            f"{r.experiment_id}/{quote(r.scorer_name, safe='')}",
        ),
    )
    # gateway_secret_permissions → (gateway_secret, <secret_id>, permission)
    _backfill_table(
        conn,
        "SELECT user_id, secret_id, permission FROM gateway_secret_permissions",
        lambda r: _resource_row(r, default, "gateway_secret", r.secret_id),
    )
    # gateway_endpoint_permissions → (gateway_endpoint, <endpoint_id>, permission)
    _backfill_table(
        conn,
        "SELECT user_id, endpoint_id, permission FROM gateway_endpoint_permissions",
        lambda r: _resource_row(r, default, "gateway_endpoint", r.endpoint_id),
    )
    # gateway_model_definition_permissions → (gateway_model_definition, <id>, permission)
    _backfill_table(
        conn,
        (
            "SELECT user_id, model_definition_id, permission "
            "FROM gateway_model_definition_permissions"
        ),
        lambda r: _resource_row(r, default, "gateway_model_definition", r.model_definition_id),
    )

    # Legacy per-resource permission tables are intentionally retained — see module
    # docstring. A future migration will drop them once the simplified RBAC model
    # has bedded in.


def downgrade() -> None:
    # The legacy per-resource permission tables were never dropped by ``upgrade()``,
    # so downgrade only needs to remove the synthetic per-user roles + their
    # role_permissions and user_role_assignments rows. The legacy tables are still
    # populated with the original pre-migration data, so operators can resume
    # using them by reverting the auth server code; no separate restore step is
    # required.
    #
    # Filter by the exact ``^__user_\d+__$`` pattern so we don't accidentally nuke
    # admin-created roles that merely start/end with underscores
    # (e.g. ``__user_admin__``). Portable via regex in Python (LIKE with ESCAPE
    # varies across dialects).
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

"""Promote ``prompt`` to a first-class ``resource_type``

Revision ID: f1a2b3c4d5e6
Revises: e5f6a7b8c9d0
Create Date: 2026-05-11 00:00:00.000000

Prompts share the model-registry wire surface with regular registered models,
so prior to this migration RBAC grants on prompts had to be expressed as
``(resource_type='registered_model', resource_pattern=<prompt-name>, permission=...)``
rows in ``role_permissions``. This was awkward because the resource_type column
could not actually distinguish the two — the discriminator only existed on the
underlying entity (the ``mlflow.prompt.is_prompt`` tag in
``registered_model_tags``).

The auth layer now treats ``prompt`` as its own ``resource_type`` with its own
validators. To keep existing operator grants working after upgrade, this
migration rewrites every ``role_permissions`` row whose
``(workspace, resource_pattern)`` names a registered model that is in fact a
prompt (``mlflow.prompt.is_prompt = 'true'``) to use ``resource_type = 'prompt'``.

**Workspace-aware classification.** ``role_permissions`` rows are workspace-scoped
via ``roles.workspace``; ``registered_model_tags`` rows are also workspace-scoped
(the registered-model PK is ``(workspace, name)``). A name collision across
workspaces — e.g. ``foo`` is a prompt in ``team-a`` but a regular registered model
in ``team-b`` — must NOT cause cross-workspace mis-rewrites, so the join is keyed
on ``(workspace, name)``.

**Same-database assumption.** Like ``e5f6a7b8c9d0``, the migration assumes the
auth tables and the model-registry tables live in the same database. When the
``registered_model_tags`` table is not reachable on the current connection (the
operator runs auth on a split database), no rows are rewritten and we leave the
grants in their pre-migration ``registered_model`` shape. Operators on a split
deployment must run an equivalent rewrite by hand — there is no robust way for
Alembic to classify rows across two databases.

**Wildcard rows** (``resource_pattern = '*'``) are left untouched. A wildcard
``registered_model`` grant covers every registered model in the user's
workspace and was never prompt-specific; rewriting it to ``prompt`` would
silently revoke registered-model access. Operators who want a wildcard prompt
grant must add it explicitly after upgrade.

"""

from alembic import op
from sqlalchemy import bindparam, inspect, text

_IS_PROMPT_TAG_KEY = "mlflow.prompt.is_prompt"

revision = "f1a2b3c4d5e6"
down_revision = "e5f6a7b8c9d0"
branch_labels = None
depends_on = None


def _registry_tag_table_exists(conn) -> bool:
    return "registered_model_tags" in inspect(conn).get_table_names()


def _classify_prompts(conn, pairs: list[tuple[str, str]]) -> set[tuple[str, str]]:
    """Return the subset of ``(workspace, name)`` pairs whose ``is_prompt`` tag is ``'true'``.

    Classification is keyed on both columns because the same name can be a
    registered model in one workspace and a prompt in another — collapsing to
    name-only would mis-rewrite grants across workspaces.
    """
    if not pairs:
        return set()
    workspaces = list({workspace for workspace, _ in pairs})
    names = list({name for _, name in pairs})
    rows = conn.execute(
        text(
            "SELECT workspace, name FROM registered_model_tags "
            "WHERE key = :key AND LOWER(value) = 'true' "
            "AND workspace IN :workspaces AND name IN :names"
        ).bindparams(
            bindparam("workspaces", expanding=True),
            bindparam("names", expanding=True),
        ),
        {"key": _IS_PROMPT_TAG_KEY, "workspaces": workspaces, "names": names},
    )
    prompt_pairs = {(row.workspace, row.name) for row in rows}
    return prompt_pairs & set(pairs)


def upgrade() -> None:
    conn = op.get_bind()
    if not _registry_tag_table_exists(conn):
        # Split-DB deployment: cannot classify across databases. See module docstring.
        return

    candidates = conn.execute(
        text(
            "SELECT rp.id, rp.resource_pattern, r.workspace "
            "FROM role_permissions rp "
            "JOIN roles r ON r.id = rp.role_id "
            "WHERE rp.resource_type = 'registered_model' AND rp.resource_pattern <> '*'"
        )
    ).fetchall()
    if not candidates:
        return

    prompt_pairs = _classify_prompts(
        conn, [(row.workspace, row.resource_pattern) for row in candidates]
    )
    if not prompt_pairs:
        return

    rewrite_ids = [
        row.id for row in candidates if (row.workspace, row.resource_pattern) in prompt_pairs
    ]
    if not rewrite_ids:
        return

    conn.execute(
        text("UPDATE role_permissions SET resource_type = 'prompt' WHERE id IN :ids").bindparams(
            bindparam("ids", expanding=True)
        ),
        {"ids": rewrite_ids},
    )


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(
        text(
            "UPDATE role_permissions SET resource_type = 'registered_model' "
            "WHERE resource_type = 'prompt'"
        )
    )

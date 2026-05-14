"""Promote ``prompt`` to a first-class ``resource_type``

Revision ID: f1a2b3c4d5e6
Revises: e5f6a7b8c9d0
Create Date: 2026-05-11 00:00:00.000000

Rewrite every ``role_permissions`` row whose ``(workspace, resource_pattern)``
names a prompt (``mlflow.prompt.is_prompt = 'true'`` in
``registered_model_tags``) from ``resource_type = 'registered_model'`` to
``resource_type = 'prompt'``. Pre-RBAC the resource_type column couldn't
distinguish prompts from regular models on the shared model-registry wire
surface; this backfills the new namespace.

* **Workspace-keyed** — the same name can be a prompt in workspace A and a
  registered model in workspace B; classifying by name alone would
  cross-rewrite grants across workspaces.
* **Wildcard rows left untouched** — ``(registered_model, *)`` covers every
  RM in the workspace and was never prompt-specific; rewriting to
  ``(prompt, *)`` would silently revoke RM access.
* **Split-DB no-op** — like ``e5f6a7b8c9d0``, assumes the auth and
  model-registry tables share a database. If ``registered_model_tags`` isn't
  reachable, nothing is rewritten; operators on a split deployment run the
  equivalent UPDATE by hand.
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
    """Return the subset of ``(workspace, name)`` pairs whose ``is_prompt`` tag is ``'true'``."""
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

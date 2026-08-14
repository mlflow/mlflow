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
from sqlalchemy import inspect, text

_IS_PROMPT_TAG_KEY = "mlflow.prompt.is_prompt"

revision = "f1a2b3c4d5e6"
down_revision = "e5f6a7b8c9d0"
branch_labels = None
depends_on = None


def _registry_tag_table_exists(conn) -> bool:
    return "registered_model_tags" in inspect(conn).get_table_names()


def upgrade() -> None:
    conn = op.get_bind()
    if not _registry_tag_table_exists(conn):
        # Split-DB deployment: cannot classify across databases. See module docstring.
        return

    # Single-pass UPDATE: rewrite every non-wildcard ``registered_model`` row whose
    # ``(workspace, name)`` matches an ``is_prompt='true'`` tag in the registry.
    # Workspace correlation goes via ``role_permissions.role_id → roles.workspace``;
    # the EXISTS subquery keeps the work in the database so we don't pull rows into
    # Python or build IN-lists that can blow past SQLite's variable limit on large
    # auth stores.
    conn.execute(
        text(
            "UPDATE role_permissions SET resource_type = 'prompt' "
            "WHERE resource_type = 'registered_model' AND resource_pattern <> '*' "
            "AND EXISTS ("
            "  SELECT 1 FROM roles r, registered_model_tags rmt "
            "  WHERE r.id = role_permissions.role_id "
            "  AND rmt.workspace = r.workspace "
            "  AND rmt.name = role_permissions.resource_pattern "
            "  AND rmt.key = :key AND LOWER(rmt.value) = 'true'"
            ")"
        ),
        {"key": _IS_PROMPT_TAG_KEY},
    )


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(
        text(
            "UPDATE role_permissions SET resource_type = 'registered_model' "
            "WHERE resource_type = 'prompt'"
        )
    )

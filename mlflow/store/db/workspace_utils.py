"""Shared helpers for workspace-related database operations.

Used by both ``workspace_migration`` (migrate-to-default-workspace) and
``workspace_move`` (move-resources).
"""

from __future__ import annotations

import sqlalchemy as sa

# Child tables of registered_models that carry their own workspace column.
# These must be updated explicitly because not all backends honour
# ON UPDATE CASCADE at the SQL level (e.g. SQLite without foreign_keys pragma).
MODEL_CHILD_TABLES = [
    "model_versions",
    "registered_model_tags",
    "model_version_tags",
    "registered_model_aliases",
]

# Tables that carry a workspace column but belong to non-model root resources.
# They must also be migrated explicitly during workspace operations.
OTHER_WORKSPACE_CHILD_TABLES = [
    "guardrail_configs",
    "mcp_server_versions",
    "mcp_server_tags",
    "mcp_server_version_tags",
    "mcp_server_aliases",
    "mcp_access_endpoints",
    "skill_versions",
    "skill_tags",
    "skill_version_tags",
    "skill_aliases",
    "agent_plugin_versions",
    "agent_plugin_tags",
    "agent_plugin_version_tags",
    "agent_plugin_aliases",
    # agent_plugin_version_members is intentionally omitted
    # (see workspace_migration._assert_no_plugin_members_outside_default).
]


_NOT_ENABLED_MSG = (
    "Aborted: the database does not have workspaces enabled. This command "
    "operates directly on the SQL database used by the default workspace "
    "provider. Please make sure the specified database URI is correct and "
    "that workspaces have been enabled via `mlflow db upgrade`."
)


def get_workspace_table(conn, table_name: str) -> sa.Table:
    """Reflect *table_name* and verify it contains a ``workspace`` column."""
    try:
        table = sa.Table(table_name, sa.MetaData(), autoload_with=conn)
    except sa.exc.NoSuchTableError:
        raise RuntimeError(f"{_NOT_ENABLED_MSG} (missing table {table_name!r}).")
    if "workspace" not in table.c:
        raise RuntimeError(_NOT_ENABLED_MSG)
    return table


def format_truncated_list(
    items: list[str],
    *,
    max_rows: int | None,
) -> str:
    """Format a list of display strings with optional truncation."""
    rows = items if max_rows is None else items[:max_rows]
    formatted = "\n  ".join(rows)
    if formatted:
        formatted = f"\n  {formatted}"
    if max_rows is not None and len(items) > max_rows:
        formatted += f"\n  ... ({len(items) - max_rows} more)"
    return formatted

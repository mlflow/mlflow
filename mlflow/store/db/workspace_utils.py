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


def validate_workspace_exists(conn, workspace_name: str) -> None:
    """Raise ``RuntimeError`` if *workspace_name* is not in the workspaces table."""
    try:
        workspaces = sa.Table("workspaces", sa.MetaData(), autoload_with=conn)
    except sa.exc.NoSuchTableError:
        raise RuntimeError(_NOT_ENABLED_MSG)
    exists = conn.execute(
        sa.select(sa.literal(1)).select_from(workspaces).where(workspaces.c.name == workspace_name)
    ).first()
    if not exists:
        raise RuntimeError(f"Workspace {workspace_name!r} does not exist.")


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

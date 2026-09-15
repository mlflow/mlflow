"""Shared helpers for workspace-related database operations.

Used by both ``workspace_migration`` (migrate-to-default-workspace) and
``workspace_move`` (move-resources).
"""

from __future__ import annotations

from contextlib import contextmanager

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
    # agent_plugin_version_members is intentionally omitted: it names its workspace
    # column ``plugin_workspace``, so it is reassigned by reassign_agent_plugin_members
    # below rather than by the generic per-table rewrite.
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


# The agent_plugin_version_members link table names its workspace column
# ``plugin_workspace`` (not ``workspace``), so it is deliberately absent from the
# workspace table lists above and is reassigned separately via the helper below.
AGENT_PLUGIN_MEMBERS_TABLE = "agent_plugin_version_members"


@contextmanager
def reassign_agent_plugin_members(
    conn,
    *,
    source_workspace: str | None,
    target_workspace: str,
    dry_run: bool = False,
):
    """Reassign ``agent_plugin_version_members`` rows to *target_workspace*.

    The link table's ``plugin_workspace`` column anchors both its FK to
    ``agent_plugin_versions`` (``ON UPDATE CASCADE``) and its FK to
    ``skill_versions`` (no cascade, ``ON DELETE NO ACTION``). A plugin version and
    its member skill versions share that single workspace value, so rewriting it
    in place while the referenced ``skill_versions`` rows are mid-move breaks the
    skill FK on every backend -- there is no statement order under per-statement
    FK checks that keeps both parents consistent with the shared column.

    Because memberships never cross workspaces, a whole-workspace reassignment always
    moves a plugin and all of its members together. This context manager captures
    and deletes the in-scope member rows on entry (before the caller moves their
    parents) and re-inserts them at *target_workspace* on exit (after both
    parents have moved), so both FKs are satisfied immediately with enforcement
    left on. It yields the number of member rows moved.

    ``source_workspace=None`` selects every row whose ``plugin_workspace`` is not
    already *target_workspace* (whole-database consolidation); a concrete value
    selects rows in that workspace only. With ``dry_run`` the rows are counted but
    left in place.
    """
    if not sa.inspect(conn).has_table(AGENT_PLUGIN_MEMBERS_TABLE):
        yield 0
        return

    table = sa.Table(AGENT_PLUGIN_MEMBERS_TABLE, sa.MetaData(), autoload_with=conn)
    column = table.c.plugin_workspace
    predicate = (
        column != target_workspace if source_workspace is None else column == source_workspace
    )

    if dry_run:
        count = conn.execute(
            sa.select(sa.func.count()).select_from(table).where(predicate)
        ).scalar_one()
        yield count
        return

    rows = [dict(row) for row in conn.execute(sa.select(table).where(predicate)).mappings()]
    if rows:
        conn.execute(sa.delete(table).where(predicate))
    yield len(rows)
    if rows:
        conn.execute(
            sa.insert(table),
            [{**row, "plugin_workspace": target_workspace} for row in rows],
        )

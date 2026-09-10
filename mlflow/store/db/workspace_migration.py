import sqlalchemy as sa

from mlflow.store.db.workspace_utils import (
    MODEL_CHILD_TABLES,
    OTHER_WORKSPACE_CHILD_TABLES,
    format_truncated_list,
    get_workspace_table,
)
from mlflow.store.workspace.sqlalchemy_store import _WORKSPACE_ROOT_MODELS
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

_WORKSPACE_TABLES = (
    [model.__tablename__ for model in _WORKSPACE_ROOT_MODELS]
    + MODEL_CHILD_TABLES
    + OTHER_WORKSPACE_CHILD_TABLES
)

_CONFLICT_SPECS = [
    ("experiments", ("name",), "experiments with the same name"),
    ("registered_models", ("name",), "registered models with the same name"),
    ("evaluation_datasets", ("name",), "evaluation datasets with the same name"),
    ("model_versions", ("name", "version"), "model versions with the same model name and version"),
    (
        "registered_model_tags",
        ("name", "key"),
        "registered model tags with the same model name and key",
    ),
    (
        "model_version_tags",
        ("name", "version", "key"),
        "model version tags with the same model name, version, and key",
    ),
    (
        "registered_model_aliases",
        ("name", "alias"),
        "registered model aliases with the same model name and alias",
    ),
    ("secrets", ("secret_name",), "secrets with the same name"),
    ("endpoints", ("name",), "endpoints with the same name"),
    ("model_definitions", ("name",), "model definitions with the same name"),
    ("mcp_servers", ("name",), "MCP servers with the same name"),
    ("skills", ("organization", "name"), "skills with the same organization and name"),
    (
        "agent_plugins",
        ("organization", "name"),
        "agent plugins with the same organization and name",
    ),
]


def _format_conflicts(
    conflicts: list[tuple[object, ...]],
    columns: tuple[str, ...],
    *,
    max_rows: int | None,
) -> str:
    display = conflicts if max_rows is None else conflicts[:max_rows]
    items = [
        ", ".join(f"{column}={value!r}" for column, value in zip(columns, row)) for row in display
    ]
    if max_rows is not None and len(conflicts) > max_rows:
        items.append(f"... ({len(conflicts) - max_rows} more)")
    return format_truncated_list(items, max_rows=None)


def _assert_no_workspace_conflicts(
    conn,
    table_name: str,
    columns: tuple[str, ...],
    resource_description: str,
    *,
    verbose: bool,
) -> None:
    table = get_workspace_table(conn, table_name)
    group_columns = [table.c[column] for column in columns]
    conflict_keys = (
        sa.select(*group_columns).group_by(*group_columns).having(sa.func.count() > 1).subquery()
    )
    join_conditions = [table.c[column] == conflict_keys.c[column] for column in columns]
    extra_columns = []
    if table_name == "experiments" and "experiment_id" in table.c:
        extra_columns.append(table.c.experiment_id)
    conflict_rows_stmt = (
        sa
        .select(*group_columns, table.c.workspace, *extra_columns)
        .select_from(table.join(conflict_keys, sa.and_(*join_conditions)))
        .order_by(*group_columns, table.c.workspace, *extra_columns)
    )
    if conflicts := conn.execute(conflict_rows_stmt).fetchall():
        formatted_conflicts = _format_conflicts(
            conflicts,
            (*columns, "workspace", *(column.name for column in extra_columns)),
            max_rows=None if verbose else 5,
        )
        raise RuntimeError(
            "Move aborted: merging workspaces would create duplicate "
            f"{resource_description}. Resolve the following conflicts by renaming the affected "
            "resources (restore deleted ones first) or permanently deleting them, then retry: "
            f"{formatted_conflicts}"
        )


# agent_plugin_version_members carries its workspace as ``plugin_workspace`` (shared
# with its skill_versions FK), not ``workspace``, so it is deliberately absent from
# _WORKSPACE_TABLES and the generic per-table loop below (which moves rows into the
# default workspace) cannot move it. Moving its parent plugins would leave every member
# row pointing at its old workspace (orphaned, or an FK failure), so migrate-to-default
# of plugin members is deferred to the workspace-lifecycle branch; until then, fail
# loudly instead of corrupting rows. See:
# https://github.com/robinnarsinghranabhat/mlflow/tree/rhaieng-7108-workspace-lifecycle
_PLUGIN_MEMBER_TABLE = "agent_plugin_version_members"


def _assert_no_plugin_members_outside_default(conn) -> None:
    try:
        table = sa.Table(_PLUGIN_MEMBER_TABLE, sa.MetaData(), autoload_with=conn)
    except sa.exc.NoSuchTableError:
        return
    count = conn.execute(
        sa
        .select(sa.func.count())
        .select_from(table)
        .where(table.c.plugin_workspace != DEFAULT_WORKSPACE_NAME)
    ).scalar_one()
    if count:
        raise RuntimeError(
            "Move aborted: migrating agent plugin members to the default workspace is not "
            f"yet supported. {count} row(s) in {_PLUGIN_MEMBER_TABLE!r} live outside the "
            f"'{DEFAULT_WORKSPACE_NAME}' workspace; moving their parent plugins would orphan "
            "them. Remove or re-home the affected agent plugin versions first, then retry."
        )


def migrate_to_default_workspace(
    engine: sa.Engine,
    dry_run: bool = False,
    *,
    verbose: bool = False,
) -> dict[str, int]:
    """
    Move all workspace-scoped resources into the default workspace.
    Returns a mapping of table name -> number of rows moved (or that would be moved in dry-run).
    When verbose is True, conflict lists are not truncated.
    """
    with engine.begin() as conn:
        # The loop below moves the parent skill and agent-plugin rows into the default
        # workspace cleanly, but it never touches agent_plugin_version_members, so those
        # member rows would be left silently pointing at their old workspace. Stop up
        # front instead of corrupting them.
        _assert_no_plugin_members_outside_default(conn)

        for table_name, columns, description in _CONFLICT_SPECS:
            _assert_no_workspace_conflicts(
                conn,
                table_name,
                columns,
                description,
                verbose=verbose,
            )

        counts = {}
        for table_name in _WORKSPACE_TABLES:
            table = get_workspace_table(conn, table_name)
            stmt = (
                sa
                .select(sa.func.count())
                .select_from(table)
                .where(table.c.workspace != DEFAULT_WORKSPACE_NAME)
            )
            counts[table_name] = conn.execute(stmt).scalar_one()

            if dry_run or counts[table_name] == 0:
                continue
            conn.execute(
                table
                .update()
                .where(table.c.workspace != DEFAULT_WORKSPACE_NAME)
                .values(workspace=DEFAULT_WORKSPACE_NAME)
            )

        return counts

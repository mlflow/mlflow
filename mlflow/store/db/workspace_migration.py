import sqlalchemy as sa

from mlflow.store.workspace.sqlalchemy_store import _WORKSPACE_ROOT_MODELS
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

# Derive table names from the shared ORM model list and add child/tags tables that also carry
# a workspace column but are not "root" tables (they are updated via FK cascades during
# delete_workspace, but the migration script must handle them explicitly).
_WORKSPACE_CHILD_TABLES = [
    "model_versions",
    "registered_model_tags",
    "model_version_tags",
    "registered_model_aliases",
]

_WORKSPACE_TABLES = [
    model.__tablename__ for model in _WORKSPACE_ROOT_MODELS
] + _WORKSPACE_CHILD_TABLES

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
]


def _format_conflicts(
    conflicts: list[tuple[object, ...]],
    columns: tuple[str, ...],
    *,
    max_rows: int | None,
) -> str:
    rows = conflicts if max_rows is None else conflicts[:max_rows]
    formatted_conflicts = "\n  ".join(
        ", ".join(f"{column}={value!r}" for column, value in zip(columns, row)) for row in rows
    )
    if formatted_conflicts:
        formatted_conflicts = f"\n  {formatted_conflicts}"
    if max_rows is not None and len(conflicts) > max_rows:
        formatted_conflicts += f"\n  ... ({len(conflicts) - max_rows} more)"
    return formatted_conflicts


def _get_table(conn, table_name: str) -> sa.Table:
    table = sa.Table(table_name, sa.MetaData(), autoload_with=conn)
    if "workspace" not in table.c:
        raise RuntimeError(
            "Move aborted: the specified tracking server does not have workspaces enabled. "
            "This command is intended for a workspace-enabled tracking server. Please make sure "
            "the specified tracking URI is correct."
        )
    return table


def _assert_no_workspace_conflicts(
    conn,
    table_name: str,
    columns: tuple[str, ...],
    resource_description: str,
    *,
    verbose: bool,
) -> None:
    table = _get_table(conn, table_name)
    group_columns = [table.c[column] for column in columns]
    conflict_keys = (
        sa.select(*group_columns).group_by(*group_columns).having(sa.func.count() > 1).subquery()
    )
    join_conditions = [table.c[column] == conflict_keys.c[column] for column in columns]
    extra_columns = []
    if table_name == "experiments" and "experiment_id" in table.c:
        extra_columns.append(table.c.experiment_id)
    conflict_rows_stmt = (
        sa.select(*group_columns, table.c.workspace, *extra_columns)
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
            table = _get_table(conn, table_name)
            stmt = (
                sa.select(sa.func.count())
                .select_from(table)
                .where(table.c.workspace != DEFAULT_WORKSPACE_NAME)
            )
            counts[table_name] = conn.execute(stmt).scalar_one()

            if dry_run or counts[table_name] == 0:
                continue
            conn.execute(
                table.update()
                .where(table.c.workspace != DEFAULT_WORKSPACE_NAME)
                .values(workspace=DEFAULT_WORKSPACE_NAME)
            )

        return counts

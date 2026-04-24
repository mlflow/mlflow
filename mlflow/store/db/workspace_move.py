from __future__ import annotations

from dataclasses import dataclass

import sqlalchemy as sa

from mlflow.store.db.workspace_utils import (
    MODEL_CHILD_TABLES,
    format_truncated_list,
    get_workspace_table,
    validate_workspace_exists,
)
from mlflow.store.model_registry.dbmodels.models import (
    SqlRegisteredModel,
    SqlRegisteredModelTag,
    SqlWebhook,
)
from mlflow.store.tracking.dbmodels.models import (
    SqlEvaluationDataset,
    SqlExperiment,
    SqlExperimentTag,
    SqlJob,
)
from mlflow.store.workspace.sqlalchemy_store import _WORKSPACE_ROOT_MODELS


@dataclass(frozen=True)
class MoveResult:
    """Result of a move-resources operation."""

    names: list[str]
    row_count: int


@dataclass(frozen=True)
class _ResourceSpec:
    """Metadata for a movable resource type."""

    model: type
    name_column: str = "name"
    tag_model: type | None = None
    # Column used to join the tag table back to the resource table.
    # For experiments the tag table joins via experiment_id, not workspace+name.
    tag_join_column: str | None = None
    child_tables: tuple[str, ...] = ()
    child_name_column: str = "name"
    has_unique_name: bool = True

    @property
    def table(self) -> sa.Table:
        return self.model.__table__

    @property
    def tag_table(self) -> sa.Table | None:
        return self.tag_model.__table__ if self.tag_model else None


# Per-model spec.  Keyed by ORM model class; the CLI resource type is derived
# from model.__tablename__ (e.g. "experiments", "registered_models").
# Models in _WORKSPACE_ROOT_MODELS without an entry here are silently skipped;
# a unit test verifies the omissions are intentional.
#
# Gateway resources (secrets, endpoints, model_definitions, budget_policies)
# are intentionally excluded because they have inter-table FK dependencies
# that make moving them independently unsafe.  They can be added later with
# proper dependency-aware handling.
_SPEC_BY_MODEL: dict[type, _ResourceSpec] = {
    SqlExperiment: _ResourceSpec(
        model=SqlExperiment,
        name_column=SqlExperiment.name.key,
        tag_model=SqlExperimentTag,
        tag_join_column=SqlExperimentTag.experiment_id.key,
    ),
    SqlRegisteredModel: _ResourceSpec(
        model=SqlRegisteredModel,
        name_column=SqlRegisteredModel.name.key,
        tag_model=SqlRegisteredModelTag,
        child_tables=tuple(MODEL_CHILD_TABLES),
    ),
    SqlEvaluationDataset: _ResourceSpec(
        model=SqlEvaluationDataset,
        name_column=SqlEvaluationDataset.name.key,
        has_unique_name=False,
    ),
    SqlWebhook: _ResourceSpec(
        model=SqlWebhook,
        name_column=SqlWebhook.name.key,
        has_unique_name=False,
    ),
    SqlJob: _ResourceSpec(
        model=SqlJob,
        name_column=SqlJob.job_name.key,
        has_unique_name=False,
    ),
}

_RESOURCE_SPECS: dict[str, _ResourceSpec] = {
    model.__tablename__: _SPEC_BY_MODEL[model]
    for model in _WORKSPACE_ROOT_MODELS
    if model in _SPEC_BY_MODEL
}
RESOURCE_TYPE_CHOICES = sorted(_RESOURCE_SPECS)


def _tag_names_subquery(
    spec: _ResourceSpec,
    source_workspace: str,
    tags: list[tuple[str, str]],
) -> sa.Select:
    """Build a SELECT subquery of resource names matching ALL given tags.

    The intersection logic uses ``GROUP BY … HAVING COUNT`` so the entire
    resolution stays in SQL and avoids materializing a large parameter list.
    """
    table = spec.table
    tag_table = spec.tag_table
    unique_tags = list(dict.fromkeys(tags))

    tag_conditions = sa.or_(*[
        sa.and_(tag_table.c.key == k, tag_table.c.value == v) for k, v in unique_tags
    ])

    if spec.tag_join_column:
        # Experiments: tags reference the resource via a surrogate key
        # (experiment_id), so we JOIN back to the resource table to get
        # the name and scope by the resource table's workspace column.
        id_col = spec.tag_join_column
        name_col = table.c[spec.name_column]
        subq = (
            sa
            .select(name_col)
            .select_from(table.join(tag_table, table.c[id_col] == tag_table.c[id_col]))
            .where(table.c.workspace == source_workspace)
            .where(tag_conditions)
            .group_by(name_col)
        )
    else:
        # Registered models: the tag table carries workspace + name
        # directly, so we can query it without joining the parent table.
        name_col = tag_table.c[spec.name_column]
        subq = (
            sa
            .select(name_col)
            .where(tag_table.c.workspace == source_workspace)
            .where(tag_conditions)
            .group_by(name_col)
        )

    if len(unique_tags) > 1:
        subq = subq.having(sa.func.count() == len(unique_tags))

    return subq


def _resolve_names(
    conn,
    spec: _ResourceSpec,
    workspace: str,
    names: list[str] | None = None,
) -> set[str]:
    """Return resource names in *workspace*, optionally filtered to *names*."""
    table = spec.table
    name_col = table.c[spec.name_column]
    stmt = sa.select(name_col).where(table.c.workspace == workspace)
    if names is not None:
        stmt = stmt.where(name_col.in_(names))
    return {row[0] for row in conn.execute(stmt).fetchall()}


def _find_conflicts(
    conn,
    spec: _ResourceSpec,
    source_workspace: str,
    target_workspace: str,
    name_filter: list[str] | sa.Select | None = None,
) -> list[str]:
    """Return source resource names that already exist in *target_workspace*.

    *name_filter* can be a ``list`` (literal names), a ``Select`` subquery,
    or ``None`` (move-all, falls back to a source-workspace subquery).
    SQLAlchemy's ``in_()`` handles both lists and subqueries transparently.
    """
    if not spec.has_unique_name:
        return []
    table = spec.table
    name_col = table.c[spec.name_column]
    stmt = sa.select(name_col).where(table.c.workspace == target_workspace)

    if name_filter is not None:
        stmt = stmt.where(name_col.in_(name_filter))
    else:
        source_subq = sa.select(name_col).where(table.c.workspace == source_workspace)
        stmt = stmt.where(name_col.in_(source_subq))

    return [row[0] for row in conn.execute(stmt.order_by(name_col)).fetchall()]


def move_resources(
    engine: sa.Engine,
    source_workspace: str,
    target_workspace: str,
    resource_type: str,
    names: list[str] | None = None,
    tags: list[tuple[str, str]] | None = None,
    dry_run: bool = False,
    *,
    verbose: bool = False,
) -> MoveResult:
    """
    Move resources of *resource_type* from *source_workspace* to *target_workspace*.

    Filter by *names* or *tags* (mutually exclusive).  When neither is provided
    all resources of the type in the source workspace are moved.

    Returns a :class:`MoveResult` with ``names`` (sorted list of distinct
    resource names that were moved or would be moved) and ``row_count`` (the
    number of rows in the root resource table that were moved; child-table
    rows such as model versions or tags are not included in this count).
    For resource types whose names are not unique, ``row_count`` may exceed
    ``len(names)`` when multiple rows share the same name.
    """
    if source_workspace == target_workspace:
        raise RuntimeError("Source and target workspaces must be different.")

    spec = _RESOURCE_SPECS.get(resource_type)
    if spec is None:
        raise RuntimeError(
            f"Unknown resource type {resource_type!r}. "
            f"Valid types: {', '.join(RESOURCE_TYPE_CHOICES)}"
        )

    if names and tags:
        raise RuntimeError("--name and --tag are mutually exclusive.")

    if tags and spec.tag_table is None:
        raise RuntimeError(f"Resource type {resource_type!r} does not support tag filtering.")

    with engine.begin() as conn:
        validate_workspace_exists(conn, source_workspace)
        validate_workspace_exists(conn, target_workspace)

        # Fail fast with a clear message if the resource table lacks a
        # workspace column (DB not migrated to workspace-enabled schema).
        get_workspace_table(conn, spec.table.name)

        # Build a unified name filter: a SQL subquery (--tag), a small
        # literal list (--name), or None (move-all).  SQLAlchemy's in_()
        # handles lists and Select objects identically, so every subsequent
        # query uses the same one-branch pattern.
        if tags:
            name_filter = _tag_names_subquery(spec, source_workspace, tags)
            matched = {row[0] for row in conn.execute(name_filter).fetchall()}
        elif names:
            matched = _resolve_names(conn, spec, source_workspace, names)
            name_filter = list(matched)
        else:
            matched = _resolve_names(conn, spec, source_workspace)
            name_filter = None

        if not matched:
            return MoveResult(names=[], row_count=0)

        if conflicts := _find_conflicts(
            conn, spec, source_workspace, target_workspace, name_filter
        ):
            formatted = format_truncated_list(
                [repr(name) for name in conflicts],
                max_rows=None if verbose else 10,
            )
            raise RuntimeError(
                f"Move aborted: the following {resource_type} already exist "
                f"in workspace {target_workspace!r} and would conflict: "
                f"{formatted}\n"
                "Rename or remove the conflicting resources in the target "
                "workspace, then retry."
            )

        table = spec.table
        name_col = table.c[spec.name_column]

        def _filtered(stmt, col, _nf=name_filter):
            return stmt.where(col.in_(_nf)) if _nf is not None else stmt

        row_count = conn.execute(
            _filtered(
                sa
                .select(sa.func.count())
                .select_from(table)
                .where(table.c.workspace == source_workspace),
                name_col,
            )
        ).scalar()

        if not dry_run:
            conn.execute(
                _filtered(
                    table
                    .update()
                    .where(table.c.workspace == source_workspace)
                    .values(workspace=target_workspace),
                    name_col,
                )
            )

            # Explicitly update child tables because not all backends honour
            # ON UPDATE CASCADE (e.g. SQLite without the foreign_keys pragma).
            for child_table_name in spec.child_tables:
                child = get_workspace_table(conn, child_table_name)
                conn.execute(
                    _filtered(
                        child
                        .update()
                        .where(child.c.workspace == source_workspace)
                        .values(workspace=target_workspace),
                        child.c[spec.child_name_column],
                    )
                )

    return MoveResult(names=sorted(matched), row_count=row_count)

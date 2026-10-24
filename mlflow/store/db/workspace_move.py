from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import sqlalchemy as sa

from mlflow.store.db.workspace_utils import (
    MODEL_CHILD_TABLES,
    format_truncated_list,
    get_workspace_table,
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
    SqlMCPServer,
    SqlMCPServerTag,
    SqlSkill,
    SqlSkillTag,
)
from mlflow.store.workspace.abstract_store import AbstractStore
from mlflow.store.workspace.sqlalchemy_store import _WORKSPACE_ROOT_MODELS
from mlflow.utils.uri import append_to_uri_path
from mlflow.utils.workspace_utils import WORKSPACES_DIR_NAME


@dataclass(frozen=True)
class MoveResult:
    """Result of a move-resources operation."""

    names: list[str]
    row_count: int
    # Set when artifact_policy="retarget": the artifact root the moved
    # experiments were repointed under.
    retarget_root: str | None = None


@dataclass(frozen=True)
class _ResourceSpec:
    """Metadata for a movable resource type."""

    model: type
    name_column: str = "name"
    tag_model: type | None = None
    # Column used to join the tag table back to the resource table.
    # For experiments the tag table joins via experiment_id, not workspace+name.
    tag_join_column: str | None = None
    child_tables: tuple[str | tuple[str, str], ...] = ()
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
    SqlMCPServer: _ResourceSpec(
        model=SqlMCPServer,
        name_column=SqlMCPServer.name.key,
        tag_model=SqlMCPServerTag,
        child_tables=(
            "mcp_server_versions",
            "mcp_server_tags",
            "mcp_server_version_tags",
            "mcp_server_aliases",
            ("mcp_access_endpoints", "server_name"),
        ),
    ),
    SqlSkill: _ResourceSpec(
        model=SqlSkill,
        name_column=SqlSkill.name.key,
        tag_model=SqlSkillTag,
        child_tables=(
            "skill_versions",
            "skill_tags",
            "skill_version_tags",
            "skill_aliases",
        ),
    ),
}

_RESOURCE_SPECS: dict[str, _ResourceSpec] = {
    model.__tablename__: _SPEC_BY_MODEL[model]
    for model in _WORKSPACE_ROOT_MODELS
    if model in _SPEC_BY_MODEL
}
RESOURCE_TYPE_CHOICES = sorted(_RESOURCE_SPECS)


def _parse_skill_name(raw_name: str) -> tuple[str, str]:
    """Parse a skill identifier into (organization, name).

    Accepts '@org/name' or 'name'.
    """
    if raw_name.startswith("@"):
        org, sep, name = raw_name[1:].partition("/")
        if sep and org and name:
            return org, name
    return "", raw_name


def _format_skill_name(organization: str, name: str) -> str:
    """Format a skill's organization and name for display and CLI output."""
    return f"@{organization}/{name}" if organization else name


def _skill_filter_condition(
    table: sa.Table, identities: list[tuple[str, str]]
) -> sa.BinaryExpression:
    """Build a dialect-agnostic WHERE condition matching (organization, name) pairs."""
    names_by_org: dict[str, set[str]] = {}
    for org, name in identities:
        names_by_org.setdefault(org, set()).add(name)
    clauses = [
        sa.and_(table.c.organization == org, table.c.name.in_(sorted(names)))
        for org, names in names_by_org.items()
    ]
    return sa.or_(*clauses) if clauses else sa.false()


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
    elif spec.model is SqlSkill:
        subq = (
            sa
            .select(tag_table.c.organization, tag_table.c.name)
            .where(tag_table.c.workspace == source_workspace)
            .where(tag_conditions)
            .group_by(tag_table.c.organization, tag_table.c.name)
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
    if spec.model is SqlSkill:
        stmt = sa.select(table.c.organization, table.c.name).where(table.c.workspace == workspace)
        if names is not None:
            identities = [_parse_skill_name(n) for n in names]
            stmt = stmt.where(_skill_filter_condition(table, identities))
        return {_format_skill_name(row[0], row[1]) for row in conn.execute(stmt).fetchall()}

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
    name_filter: list[str] | list[tuple[str, str]] | sa.Select | None = None,
) -> list[str]:
    """Return source resource names that already exist in *target_workspace*.

    *name_filter* can be a ``list`` (literal names or (org, name) pairs for skills),
    a ``Select`` subquery, or ``None`` (move-all, falls back to a source-workspace subquery).
    SQLAlchemy's ``in_()`` handles both lists and subqueries transparently.
    """
    if not spec.has_unique_name:
        return []
    table = spec.table
    if spec.model is SqlSkill:
        stmt = sa.select(table.c.organization, table.c.name).where(
            table.c.workspace == target_workspace
        )
        if name_filter is not None:
            stmt = stmt.where(_skill_filter_condition(table, name_filter))
        else:
            source_table = table.alias("source_skills")
            stmt = stmt.where(
                sa.exists(
                    sa.select(1).where(
                        source_table.c.workspace == source_workspace,
                        source_table.c.organization == table.c.organization,
                        source_table.c.name == table.c.name,
                    )
                )
            )
        rows = conn.execute(stmt.order_by(table.c.organization, table.c.name)).fetchall()
        return [_format_skill_name(org, name) for org, name in rows]

    name_col = table.c[spec.name_column]
    stmt = sa.select(name_col).where(table.c.workspace == target_workspace)

    if name_filter is not None:
        stmt = stmt.where(name_col.in_(name_filter))
    else:
        source_subq = sa.select(name_col).where(table.c.workspace == source_workspace)
        stmt = stmt.where(name_col.in_(source_subq))

    return [row[0] for row in conn.execute(stmt.order_by(name_col)).fetchall()]


def _resolve_target_artifact_root(
    workspace_store: AbstractStore,
    target_workspace: str,
    default_artifact_root: str | None,
) -> str:
    """Resolve the artifact root for the target workspace via the workspace provider.

    Mirrors the tracking server's resolution when creating experiments: a
    workspace-level default_artifact_root is used as is, and the server-level
    root gets the workspaces/<name> suffix appended.
    """
    root, append_workspace_prefix = workspace_store.resolve_artifact_root(
        default_artifact_root, target_workspace
    )
    if not root:
        raise RuntimeError(
            f"Cannot determine the artifact root for workspace {target_workspace!r}. "
            "Pass --default-artifact-root with the value the tracking server is "
            "started with, or configure the workspace's default_artifact_root."
        )
    if append_workspace_prefix:
        root = append_to_uri_path(root, WORKSPACES_DIR_NAME, target_workspace)
    return root


def move_resources(
    engine: sa.Engine,
    workspace_store: AbstractStore,
    source_workspace: str,
    target_workspace: str,
    resource_type: str,
    names: list[str] | None = None,
    tags: list[tuple[str, str]] | None = None,
    dry_run: bool = False,
    *,
    verbose: bool = False,
    artifact_policy: Literal["preserve", "retarget"] = "preserve",
    default_artifact_root: str | None = None,
) -> MoveResult:
    """
    Move resources of *resource_type* from *source_workspace* to *target_workspace*.

    Filter by *names* or *tags* (mutually exclusive).  When neither is provided
    all resources of the type in the source workspace are moved.

    Workspace validation and artifact root resolution go through
    *workspace_store*, which is not necessarily backed by the tracking database
    this command operates on.

    With ``artifact_policy="retarget"`` (experiments only), every moved
    experiment's ``artifact_location`` is repointed to the artifact root
    resolved for the target workspace, in the same transaction as the move.
    Artifact objects and stored run, logged model and trace URIs are not
    modified.

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

    if artifact_policy not in ("preserve", "retarget"):
        raise RuntimeError(f"Unknown artifact policy {artifact_policy!r}.")

    if artifact_policy == "retarget" and resource_type != "experiments":
        raise RuntimeError(
            "--artifact-policy retarget is only supported for --resource-type experiments."
        )

    workspace_store.get_workspace(source_workspace)
    workspace_store.get_workspace(target_workspace)

    retarget_root = None
    if artifact_policy == "retarget":
        retarget_root = _resolve_target_artifact_root(
            workspace_store, target_workspace, default_artifact_root
        )

    with engine.begin() as conn:
        # Fail fast with a clear message if the resource table lacks a
        # workspace column (DB not migrated to workspace-enabled schema).
        get_workspace_table(conn, spec.table.name)

        # Build a unified name filter: a SQL subquery (--tag), a small
        # literal list (--name), or None (move-all).  SQLAlchemy's in_()
        # handles lists and Select objects identically, so every subsequent
        # query uses the same one-branch pattern.
        if tags:
            tag_subq = _tag_names_subquery(spec, source_workspace, tags)
            if spec.model is SqlSkill:
                skill_rows = conn.execute(tag_subq).fetchall()
                name_filter = [(row[0], row[1]) for row in skill_rows]
                matched = {_format_skill_name(org, name) for org, name in name_filter}
            else:
                name_filter = tag_subq
                matched = {row[0] for row in conn.execute(name_filter).fetchall()}
        elif names:
            matched = _resolve_names(conn, spec, source_workspace, names)
            if spec.model is SqlSkill:
                name_filter = [_parse_skill_name(n) for n in matched]
            else:
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

        if spec.model is SqlSkill:
            count_stmt = (
                sa
                .select(sa.func.count())
                .select_from(table)
                .where(table.c.workspace == source_workspace)
            )
            if name_filter is not None:
                count_stmt = count_stmt.where(_skill_filter_condition(table, name_filter))
            row_count = conn.execute(count_stmt).scalar()
        else:
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
            if retarget_root is not None:
                # Resolved before the flip because the tag-filter subquery in
                # name_filter is scoped to the source workspace. Locations are
                # built with append_to_uri_path to match how the server derives
                # them when creating experiments.
                retarget_params = [
                    {
                        "pk": experiment_id,
                        "new_root": append_to_uri_path(retarget_root, str(experiment_id)),
                    }
                    for (experiment_id,) in conn.execute(
                        _filtered(
                            sa.select(table.c.experiment_id).where(
                                table.c.workspace == source_workspace
                            ),
                            name_col,
                        )
                    )
                ]
                conn.execute(
                    table
                    .update()
                    .where(table.c.experiment_id == sa.bindparam("pk"))
                    .values(artifact_location=sa.bindparam("new_root")),
                    retarget_params,
                )

            if spec.model is SqlSkill:
                table_upd = (
                    table
                    .update()
                    .where(table.c.workspace == source_workspace)
                    .values(workspace=target_workspace)
                )
                if name_filter is not None:
                    table_upd = table_upd.where(_skill_filter_condition(table, name_filter))
                conn.execute(table_upd)
            else:
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
            # Each entry is either a table name str (uses spec.child_name_column)
            # or a (table_name, column_name) tuple for non-standard FK columns.
            for entry in spec.child_tables:
                if isinstance(entry, tuple):
                    child_table_name = entry[0]
                    col_name = entry[1]
                else:
                    child_table_name = entry
                    col_name = spec.child_name_column
                child = get_workspace_table(conn, child_table_name)
                child_upd = (
                    child
                    .update()
                    .where(child.c.workspace == source_workspace)
                    .values(workspace=target_workspace)
                )
                if spec.model is SqlSkill:
                    if name_filter is not None:
                        child_upd = child_upd.where(_skill_filter_condition(child, name_filter))
                else:
                    child_upd = _filtered(child_upd, child.c[col_name])
                conn.execute(child_upd)

    return MoveResult(names=sorted(matched), row_count=row_count, retarget_root=retarget_root)

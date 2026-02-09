"""Add workspace columns and catalog table

Create Date: 2026-01-16 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = "1b5f0d9ad7c1"
down_revision = "c8d9e0f1a2b3"
branch_labels = None
depends_on = None

_NAMING_CONVENTION = {
    "pk": "pk_%(table_name)s",
    "fk": "fk_%(table_name)s_%(referred_table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
}

_WORKSPACE_TABLES = [
    "experiments",
    "registered_models",
    "model_versions",
    "registered_model_tags",
    "model_version_tags",
    "registered_model_aliases",
    "evaluation_datasets",
    "webhooks",
    "secrets",
    "endpoints",
    "model_definitions",
    "jobs",
]

# SQLite trigger to enforce immutability of secret_id and secret_name after batch table rebuilds.
_SQLITE_SECRETS_IMMUTABILITY_TRIGGER = """
CREATE TRIGGER prevent_secrets_aad_mutation
BEFORE UPDATE ON secrets
FOR EACH ROW
WHEN OLD.secret_id != NEW.secret_id OR OLD.secret_name != NEW.secret_name
BEGIN
    SELECT RAISE(ABORT, 'secret_id and secret_name are immutable (used as AAD in encryption)');
END;
"""


def _recreate_secrets_immutability_trigger(dialect_name: str) -> None:
    if dialect_name == "sqlite":
        op.execute("DROP TRIGGER IF EXISTS prevent_secrets_aad_mutation;")
        op.execute(_SQLITE_SECRETS_IMMUTABILITY_TRIGGER)


# Older SQLite migrations emitted unnamed foreign keys. When batch-altering tables we need the
# legacy names so we can drop the constraints deterministically; this mapping gives us the
# aliases for those historical definitions.
_SQLITE_LEGACY_FKS = {
    (
        "model_versions",
        "registered_models",
        ("name",),
    ): "fk_model_versions_registered_models_name",
    (
        "registered_model_tags",
        "registered_models",
        ("name",),
    ): "fk_registered_model_tags_registered_models_name",
    (
        "model_version_tags",
        "model_versions",
        ("name", "version"),
    ): "fk_model_version_tags_model_versions_name",
    (
        "registered_model_aliases",
        "registered_models",
        ("name",),
    ): "fk_registered_model_aliases_registered_models_name",
}


def _workspace_column():
    return sa.Column(
        "workspace",
        sa.String(length=63),
        nullable=False,
        server_default=sa.text("'default'"),
    )


def _fetch_mssql_unique_metadata(
    conn, dialect_name: str, schema: str | None, table_name: str, for_indexes: bool = False
):
    """Fetch unique constraint or index metadata for the given MSSQL table.

    SQLAlchemy's inspector doesn't implement ``get_unique_constraints`` for MSSQL, so we query the
    catalog tables directly. When ``for_indexes`` is True we return unique indexes (needed when a
    unique constraint is materialized as an index); otherwise we return metadata that looks like
    ``get_unique_constraints``.
    """
    if dialect_name != "mssql":
        return []

    if for_indexes:
        query = sa.text(
            """
            SELECT
                i.name,
                STRING_AGG(c.name, ',') WITHIN GROUP (ORDER BY ic.key_ordinal) AS column_names
            FROM sys.indexes i
            JOIN sys.tables t ON i.object_id = t.object_id
            JOIN sys.schemas s ON t.schema_id = s.schema_id
            JOIN sys.index_columns ic
                ON i.object_id = ic.object_id
               AND i.index_id = ic.index_id
            JOIN sys.columns c
                ON ic.object_id = c.object_id
               AND ic.column_id = c.column_id
            WHERE i.is_unique = 1
              AND i.is_primary_key = 0
              AND ic.is_included_column = 0
              AND t.name = :table_name
              AND (:schema IS NULL OR s.name = :schema)
            GROUP BY i.name
            """
        )
    else:
        query = sa.text(
            """
            SELECT
                kc.name,
                STRING_AGG(c.name, ',') WITHIN GROUP (ORDER BY ic.key_ordinal) AS column_names
            FROM sys.key_constraints kc
            JOIN sys.tables t ON kc.parent_object_id = t.object_id
            JOIN sys.schemas s ON t.schema_id = s.schema_id
            JOIN sys.index_columns ic
                ON kc.parent_object_id = ic.object_id
               AND kc.unique_index_id = ic.index_id
            JOIN sys.columns c
                ON ic.object_id = c.object_id
               AND ic.column_id = c.column_id
            WHERE kc.type = 'UQ'
              AND t.name = :table_name
              AND (:schema IS NULL OR s.name = :schema)
            GROUP BY kc.name
            """
        )

    result = conn.execute(query, {"table_name": table_name, "schema": schema}).fetchall()
    return [
        {
            "name": row[0],
            "column_names": [col.strip() for col in row[1].split(",") if col] if row[1] else [],
        }
        for row in result
    ]


def _with_batch(table_name):
    return op.batch_alter_table(table_name, recreate="auto", naming_convention=_NAMING_CONVENTION)


# Once the tracking and model registry stores support workspaces, we can remove the
# server_default to ensure the stores are properly setting the workspace.
def upgrade():
    conn = op.get_bind()
    inspector = inspect(conn)
    dialect_name = conn.dialect.name
    schema = op.get_context().version_table_schema or inspector.default_schema_name

    def _get_unique_constraints(table_name: str):
        try:
            return inspector.get_unique_constraints(table_name)
        except NotImplementedError:
            if dialect_name != "mssql":
                raise
            # SQL Server's inspector does not implement get_unique_constraints; fall back to
            # querying the catalog tables directly via _fetch_mssql_unique_metadata.
            return _fetch_mssql_unique_metadata(
                conn,
                dialect_name,
                schema,
                table_name,
                for_indexes=False,
            )

    def _get_unique_indexes(table_name: str):
        try:
            return inspector.get_indexes(table_name)
        except NotImplementedError:
            if dialect_name != "mssql":
                raise

            metadata = (
                _fetch_mssql_unique_metadata(
                    conn,
                    dialect_name,
                    schema,
                    table_name,
                    for_indexes=True,
                )
                or []
            )

            for entry in metadata:
                entry["unique"] = True
            return metadata

    def _detect_unique_on_column(table_name: str, column_name: str = "name"):
        """Detect unique constraint or index on a specific column."""
        expected_name = _NAMING_CONVENTION["uq"] % {
            "table_name": table_name,
            "column_0_name": column_name,
        }

        for constraint in _get_unique_constraints(table_name) or []:
            cols = constraint.get("column_names") or []
            name = constraint.get("name")
            if cols == [column_name] or name == expected_name:
                return name or expected_name, None

        for index in _get_unique_indexes(table_name) or []:
            if index.get("unique") and index.get("column_names") == [column_name]:
                return None, index["name"]

        return None, None

    def _collect_foreign_keys(table: str, referred_table: str):
        names = []
        for fk in inspector.get_foreign_keys(table):
            if fk.get("referred_table") == referred_table:
                name = fk.get("name")
                if not name and dialect_name == "sqlite":
                    key = (table, referred_table, tuple(fk.get("constrained_columns") or ()))
                    name = _SQLITE_LEGACY_FKS.get(key)
                    if not name:
                        continue
                names.append(name)
        return names

    def _create_workspace_indexes_and_catalog():
        op.create_index("idx_experiments_workspace", "experiments", ["workspace"])
        op.create_index("idx_registered_models_workspace", "registered_models", ["workspace"])
        op.create_index(
            "idx_experiments_workspace_creation_time",
            "experiments",
            ["workspace", "creation_time"],
            unique=False,
        )
        op.create_index("idx_evaluation_datasets_workspace", "evaluation_datasets", ["workspace"])
        op.create_index("idx_webhooks_workspace", "webhooks", ["workspace"])
        op.create_index("idx_secrets_workspace", "secrets", ["workspace"])
        op.create_index("idx_endpoints_workspace", "endpoints", ["workspace"])
        op.create_index("idx_model_definitions_workspace", "model_definitions", ["workspace"])

        op.create_table(
            "workspaces",
            sa.Column("name", sa.String(length=63), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("default_artifact_root", sa.Text(), nullable=True),
            sa.PrimaryKeyConstraint("name", name="workspaces_pk"),
        )

        metadata = sa.MetaData()
        workspaces_table = sa.Table(
            "workspaces",
            metadata,
            sa.Column("name", sa.String(length=63)),
            sa.Column("description", sa.Text()),
            sa.Column("default_artifact_root", sa.Text()),
        )

        conn.execute(
            workspaces_table.insert().values(
                name="default",
                description="Default workspace for legacy resources",
            )
        )

    experiments_unique_constraint, experiments_unique_index = _detect_unique_on_column(
        "experiments"
    )
    registered_models_unique_constraint, registered_models_unique_index = _detect_unique_on_column(
        "registered_models"
    )
    secrets_unique_constraint, secrets_unique_index = _detect_unique_on_column(
        "secrets", "secret_name"
    )
    endpoints_unique_constraint, endpoints_unique_index = _detect_unique_on_column("endpoints")
    model_definitions_unique_constraint, model_definitions_unique_index = _detect_unique_on_column(
        "model_definitions"
    )

    fk_model_versions = _collect_foreign_keys("model_versions", "registered_models")
    fk_registered_model_tags = _collect_foreign_keys("registered_model_tags", "registered_models")
    fk_registered_model_aliases = _collect_foreign_keys(
        "registered_model_aliases", "registered_models"
    )
    fk_model_version_tags = _collect_foreign_keys("model_version_tags", "model_versions")

    if dialect_name == "sqlite":
        # Let SQLite handle the foreign key drops inside the batches so each table is recreated
        # only once.
        with _with_batch("experiments") as batch_op:
            if experiments_unique_constraint:
                batch_op.drop_constraint(experiments_unique_constraint, type_="unique")
            elif experiments_unique_index:
                batch_op.drop_index(experiments_unique_index)
            batch_op.add_column(_workspace_column())
            batch_op.create_unique_constraint(
                "uq_experiments_workspace_name",
                ["workspace", "name"],
            )

        with _with_batch("registered_models") as batch_op:
            if registered_models_unique_constraint:
                batch_op.drop_constraint(registered_models_unique_constraint, type_="unique")
            elif registered_models_unique_index:
                batch_op.drop_index(registered_models_unique_index)
            batch_op.add_column(_workspace_column())
            batch_op.drop_constraint("registered_model_pk", type_="primary")
            batch_op.create_primary_key("registered_model_pk", ["workspace", "name"])

        with _with_batch("model_versions") as batch_op:
            batch_op.add_column(_workspace_column())
            for fk_name in fk_model_versions:
                batch_op.drop_constraint(fk_name, type_="foreignkey")
            batch_op.drop_constraint("model_version_pk", type_="primary")
            batch_op.create_primary_key("model_version_pk", ["workspace", "name", "version"])
            batch_op.create_foreign_key(
                "fk_model_versions_registered_models",
                "registered_models",
                ["workspace", "name"],
                ["workspace", "name"],
                onupdate="CASCADE",
            )

        with _with_batch("registered_model_tags") as batch_op:
            batch_op.add_column(_workspace_column())
            for fk_name in fk_registered_model_tags:
                batch_op.drop_constraint(fk_name, type_="foreignkey")
            batch_op.drop_constraint("registered_model_tag_pk", type_="primary")
            batch_op.create_primary_key("registered_model_tag_pk", ["workspace", "key", "name"])
            batch_op.create_foreign_key(
                "fk_registered_model_tags_registered_models",
                "registered_models",
                ["workspace", "name"],
                ["workspace", "name"],
                onupdate="CASCADE",
            )

        with _with_batch("model_version_tags") as batch_op:
            batch_op.add_column(_workspace_column())
            for fk_name in fk_model_version_tags:
                batch_op.drop_constraint(fk_name, type_="foreignkey")
            batch_op.drop_constraint("model_version_tag_pk", type_="primary")
            batch_op.create_primary_key(
                "model_version_tag_pk",
                ["workspace", "key", "name", "version"],
            )
            batch_op.create_foreign_key(
                "fk_model_version_tags_model_versions",
                "model_versions",
                ["workspace", "name", "version"],
                ["workspace", "name", "version"],
                onupdate="CASCADE",
            )

        with _with_batch("registered_model_aliases") as batch_op:
            batch_op.add_column(_workspace_column())
            for fk_name in fk_registered_model_aliases:
                batch_op.drop_constraint(fk_name, type_="foreignkey")
            batch_op.drop_constraint("registered_model_alias_pk", type_="primary")
            batch_op.create_primary_key(
                "registered_model_alias_pk",
                ["workspace", "name", "alias"],
            )
            batch_op.create_foreign_key(
                "fk_registered_model_aliases_registered_models",
                "registered_models",
                ["workspace", "name"],
                ["workspace", "name"],
                onupdate="CASCADE",
                ondelete="CASCADE",
            )

        with _with_batch("evaluation_datasets") as batch_op:
            batch_op.add_column(_workspace_column())

        with _with_batch("webhooks") as batch_op:
            batch_op.add_column(_workspace_column())

        with _with_batch("secrets") as batch_op:
            if secrets_unique_constraint:
                batch_op.drop_constraint(secrets_unique_constraint, type_="unique")
            elif secrets_unique_index:
                batch_op.drop_index(secrets_unique_index)
            batch_op.add_column(_workspace_column())
            batch_op.create_unique_constraint(
                "uq_secrets_workspace_secret_name",
                ["workspace", "secret_name"],
            )
        _recreate_secrets_immutability_trigger(dialect_name)

        with _with_batch("endpoints") as batch_op:
            if endpoints_unique_constraint:
                batch_op.drop_constraint(endpoints_unique_constraint, type_="unique")
            elif endpoints_unique_index:
                batch_op.drop_index(endpoints_unique_index)
            batch_op.add_column(_workspace_column())
            batch_op.create_unique_constraint(
                "uq_endpoints_workspace_name",
                ["workspace", "name"],
            )

        with _with_batch("model_definitions") as batch_op:
            if model_definitions_unique_constraint:
                batch_op.drop_constraint(model_definitions_unique_constraint, type_="unique")
            elif model_definitions_unique_index:
                batch_op.drop_index(model_definitions_unique_index)
            batch_op.add_column(_workspace_column())
            batch_op.create_unique_constraint(
                "uq_model_definitions_workspace_name",
                ["workspace", "name"],
            )

        with _with_batch("jobs") as batch_op:
            batch_op.drop_index("index_jobs_name_status_creation_time")
            batch_op.add_column(_workspace_column())
            batch_op.create_index(
                "index_jobs_name_status_creation_time",
                ["job_name", "workspace", "status", "creation_time"],
            )

        _create_workspace_indexes_and_catalog()

        return

    # Non-SQLite dialects can issue direct ALTER TABLE statements, which avoids rebuilding the
    # tables. This code duplication is worth the performance gain of not rebuilding the tables.
    # We could potentially leverage Alembic's batch_alter_table with recreate="auto" to avoid the
    # duplication, but parts of this migration caused the tables to be recreated anyways.

    def _drop_fk_constraints(table_name: str, fk_names: list[str]):
        for fk_name in fk_names:
            if fk_name:
                op.drop_constraint(fk_name, table_name=table_name, type_="foreignkey")

    def _drop_unique_on_name(table_name: str, constraint: str | None, index: str | None):
        if constraint:
            op.drop_constraint(constraint, table_name=table_name, type_="unique")
        elif index:
            op.drop_index(index, table_name=table_name)

    _drop_unique_on_name("experiments", experiments_unique_constraint, experiments_unique_index)
    op.add_column(
        "experiments",
        _workspace_column(),
    )
    op.create_unique_constraint(
        "uq_experiments_workspace_name",
        "experiments",
        ["workspace", "name"],
    )

    _drop_fk_constraints("model_versions", fk_model_versions)
    _drop_fk_constraints("registered_model_tags", fk_registered_model_tags)
    _drop_fk_constraints("registered_model_aliases", fk_registered_model_aliases)
    _drop_fk_constraints("model_version_tags", fk_model_version_tags)

    _drop_unique_on_name(
        "registered_models",
        registered_models_unique_constraint,
        registered_models_unique_index,
    )
    op.add_column(
        "registered_models",
        _workspace_column(),
    )
    op.drop_constraint("registered_model_pk", "registered_models", type_="primary")
    op.create_primary_key("registered_model_pk", "registered_models", ["workspace", "name"])

    op.add_column(
        "model_versions",
        _workspace_column(),
    )
    op.drop_constraint("model_version_pk", "model_versions", type_="primary")
    op.create_primary_key("model_version_pk", "model_versions", ["workspace", "name", "version"])
    op.create_foreign_key(
        "fk_model_versions_registered_models",
        "model_versions",
        "registered_models",
        ["workspace", "name"],
        ["workspace", "name"],
        onupdate="CASCADE",
    )

    op.add_column(
        "registered_model_tags",
        _workspace_column(),
    )
    op.drop_constraint("registered_model_tag_pk", "registered_model_tags", type_="primary")
    op.create_primary_key(
        "registered_model_tag_pk",
        "registered_model_tags",
        ["workspace", "key", "name"],
    )
    op.create_foreign_key(
        "fk_registered_model_tags_registered_models",
        "registered_model_tags",
        "registered_models",
        ["workspace", "name"],
        ["workspace", "name"],
        onupdate="CASCADE",
    )

    op.add_column(
        "model_version_tags",
        _workspace_column(),
    )
    op.drop_constraint("model_version_tag_pk", "model_version_tags", type_="primary")
    op.create_primary_key(
        "model_version_tag_pk",
        "model_version_tags",
        ["workspace", "key", "name", "version"],
    )
    op.create_foreign_key(
        "fk_model_version_tags_model_versions",
        "model_version_tags",
        "model_versions",
        ["workspace", "name", "version"],
        ["workspace", "name", "version"],
        onupdate="CASCADE",
    )

    op.add_column(
        "registered_model_aliases",
        _workspace_column(),
    )
    op.drop_constraint("registered_model_alias_pk", "registered_model_aliases", type_="primary")
    op.create_primary_key(
        "registered_model_alias_pk",
        "registered_model_aliases",
        ["workspace", "name", "alias"],
    )
    op.create_foreign_key(
        "fk_registered_model_aliases_registered_models",
        "registered_model_aliases",
        "registered_models",
        ["workspace", "name"],
        ["workspace", "name"],
        onupdate="CASCADE",
        ondelete="CASCADE",
    )

    for table in ["evaluation_datasets", "webhooks"]:
        op.add_column(
            table,
            _workspace_column(),
        )

    _drop_unique_on_name("secrets", secrets_unique_constraint, secrets_unique_index)
    op.add_column("secrets", _workspace_column())
    op.create_unique_constraint(
        "uq_secrets_workspace_secret_name",
        "secrets",
        ["workspace", "secret_name"],
    )

    _drop_unique_on_name("endpoints", endpoints_unique_constraint, endpoints_unique_index)
    op.add_column("endpoints", _workspace_column())
    op.create_unique_constraint(
        "uq_endpoints_workspace_name",
        "endpoints",
        ["workspace", "name"],
    )

    _drop_unique_on_name(
        "model_definitions", model_definitions_unique_constraint, model_definitions_unique_index
    )
    op.add_column("model_definitions", _workspace_column())
    op.create_unique_constraint(
        "uq_model_definitions_workspace_name",
        "model_definitions",
        ["workspace", "name"],
    )

    op.drop_index("index_jobs_name_status_creation_time", "jobs")
    op.add_column("jobs", _workspace_column())
    op.create_index(
        "index_jobs_name_status_creation_time",
        "jobs",
        ["job_name", "workspace", "status", "creation_time"],
    )

    _create_workspace_indexes_and_catalog()


def downgrade():
    conn = op.get_bind()
    dialect_name = conn.dialect.name

    def _assert_no_workspace_conflicts(
        table_name: str,
        columns: tuple[str, ...],
        resource_description: str,
    ):
        table = sa.Table(table_name, sa.MetaData(), autoload_with=conn)
        group_columns = [table.c[column] for column in columns]
        stmt = sa.select(*group_columns).group_by(*group_columns).having(sa.func.count() > 1)
        conflicts = conn.execute(stmt).fetchall()
        if conflicts:
            formatted_conflicts = ", ".join(
                "; ".join(f"{column}={value!r}" for column, value in zip(columns, row))
                for row in conflicts[:5]
            )
            if len(conflicts) > 5:
                formatted_conflicts += ", ..."
            raise RuntimeError(
                "Downgrade aborted: merging workspaces would create duplicate "
                f"{resource_description}. Resolve the following conflicts by deleting or renaming "
                f"the affected resources and retry: {formatted_conflicts}"
            )

    def _move_resources_to_default_workspace(table_name: str):
        table = sa.Table(table_name, sa.MetaData(), autoload_with=conn)
        conn.execute(
            table.update().where(table.c.workspace != "default").values(workspace="default")
        )

    conflict_specs = [
        ("experiments", ("name",), "experiments with the same name"),
        ("registered_models", ("name",), "registered models with the same name"),
        (
            "evaluation_datasets",
            ("name",),
            "evaluation datasets with the same name",
        ),
        (
            "model_versions",
            ("name", "version"),
            "model versions with the same model name and version",
        ),
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

    for table_name, columns, description in conflict_specs:
        _assert_no_workspace_conflicts(table_name, columns, description)

    for table in _WORKSPACE_TABLES:
        _move_resources_to_default_workspace(table)

    drop_index_kwargs = {} if dialect_name == "mysql" else {"if_exists": True}
    op.drop_index(
        "idx_experiments_workspace_creation_time", table_name="experiments", **drop_index_kwargs
    )
    op.drop_index(
        "idx_registered_models_workspace", table_name="registered_models", **drop_index_kwargs
    )
    op.drop_index("idx_experiments_workspace", table_name="experiments", **drop_index_kwargs)
    op.drop_index(
        "idx_evaluation_datasets_workspace", table_name="evaluation_datasets", **drop_index_kwargs
    )
    op.drop_index("idx_webhooks_workspace", table_name="webhooks", **drop_index_kwargs)
    op.drop_index("idx_secrets_workspace", table_name="secrets", **drop_index_kwargs)
    op.drop_index("idx_endpoints_workspace", table_name="endpoints", **drop_index_kwargs)
    op.drop_index(
        "idx_model_definitions_workspace", table_name="model_definitions", **drop_index_kwargs
    )

    if dialect_name == "sqlite":
        with _with_batch("model_version_tags") as batch_op:
            batch_op.drop_constraint("fk_model_version_tags_model_versions", type_="foreignkey")
            batch_op.drop_constraint("model_version_tag_pk", type_="primary")
            batch_op.drop_column("workspace")
            batch_op.create_primary_key("model_version_tag_pk", ["key", "name", "version"])
            batch_op.create_foreign_key(
                "model_version_tags_mv_fkey",
                "model_versions",
                ["name", "version"],
                ["name", "version"],
                onupdate="CASCADE",
            )

        with _with_batch("registered_model_aliases") as batch_op:
            batch_op.drop_constraint(
                "fk_registered_model_aliases_registered_models", type_="foreignkey"
            )
            batch_op.drop_constraint("registered_model_alias_pk", type_="primary")
            batch_op.drop_column("workspace")
            batch_op.create_primary_key("registered_model_alias_pk", ["name", "alias"])
            batch_op.create_foreign_key(
                "registered_model_alias_name_fkey",
                "registered_models",
                ["name"],
                ["name"],
                onupdate="CASCADE",
                ondelete="CASCADE",
            )

        with _with_batch("registered_model_tags") as batch_op:
            batch_op.drop_constraint(
                "fk_registered_model_tags_registered_models", type_="foreignkey"
            )
            batch_op.drop_constraint("registered_model_tag_pk", type_="primary")
            batch_op.drop_column("workspace")
            batch_op.create_primary_key("registered_model_tag_pk", ["key", "name"])
            batch_op.create_foreign_key(
                "registered_model_tags_name_fkey",
                "registered_models",
                ["name"],
                ["name"],
                onupdate="CASCADE",
            )

        with _with_batch("model_versions") as batch_op:
            batch_op.drop_constraint("fk_model_versions_registered_models", type_="foreignkey")
            batch_op.drop_constraint("model_version_pk", type_="primary")
            batch_op.drop_column("workspace")
            batch_op.create_primary_key("model_version_pk", ["name", "version"])
            batch_op.create_foreign_key(
                "model_versions_name_fkey",
                "registered_models",
                ["name"],
                ["name"],
                onupdate="CASCADE",
            )

        with _with_batch("registered_models") as batch_op:
            batch_op.drop_constraint("registered_model_pk", type_="primary")
            batch_op.drop_column("workspace")
            batch_op.create_primary_key("registered_model_pk", ["name"])

        with _with_batch("experiments") as batch_op:
            batch_op.drop_constraint("uq_experiments_workspace_name", type_="unique")
            batch_op.drop_column("workspace")
            batch_op.create_unique_constraint("uq_experiments_name", ["name"])

        with _with_batch("evaluation_datasets") as batch_op:
            batch_op.drop_column("workspace")

        with _with_batch("webhooks") as batch_op:
            batch_op.drop_column("workspace")

        with _with_batch("model_definitions") as batch_op:
            batch_op.drop_constraint("uq_model_definitions_workspace_name", type_="unique")
            batch_op.drop_column("workspace")
            batch_op.create_index("unique_model_definition_name", ["name"], unique=True)

        with _with_batch("endpoints") as batch_op:
            batch_op.drop_constraint("uq_endpoints_workspace_name", type_="unique")
            batch_op.drop_column("workspace")
            batch_op.create_index("unique_endpoint_name", ["name"], unique=True)

        with _with_batch("secrets") as batch_op:
            batch_op.drop_constraint("uq_secrets_workspace_secret_name", type_="unique")
            batch_op.drop_column("workspace")
            batch_op.create_index("unique_secret_name", ["secret_name"], unique=True)
        _recreate_secrets_immutability_trigger(dialect_name)

        with _with_batch("jobs") as batch_op:
            batch_op.drop_index("index_jobs_name_status_creation_time")
            batch_op.drop_column("workspace")
            batch_op.create_index(
                "index_jobs_name_status_creation_time",
                ["job_name", "status", "creation_time"],
            )

        op.drop_table("workspaces")
        return

    op.drop_constraint(
        "fk_model_version_tags_model_versions", "model_version_tags", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_registered_model_aliases_registered_models",
        "registered_model_aliases",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_registered_model_tags_registered_models", "registered_model_tags", type_="foreignkey"
    )
    op.drop_constraint("fk_model_versions_registered_models", "model_versions", type_="foreignkey")

    op.drop_constraint("uq_experiments_workspace_name", table_name="experiments", type_="unique")

    op.drop_constraint("registered_model_alias_pk", "registered_model_aliases", type_="primary")
    op.drop_constraint("model_version_tag_pk", "model_version_tags", type_="primary")
    op.drop_constraint("registered_model_tag_pk", "registered_model_tags", type_="primary")
    op.drop_constraint("model_version_pk", "model_versions", type_="primary")
    op.drop_constraint("registered_model_pk", "registered_models", type_="primary")
    op.drop_index("index_jobs_name_status_creation_time", "jobs")

    if dialect_name == "mssql":
        # SQL Server binds defaults via named constraints. If we try to drop the column while a
        # default is attached, the prior downgrade operations can leave behind those constraints,
        # causing drop_column to fail. Clear the defaults explicitly first.
        for table in _WORKSPACE_TABLES:
            op.alter_column(
                table_name=table,
                column_name="workspace",
                existing_type=sa.String(length=63),
                existing_nullable=False,
                server_default=None,
            )

    op.drop_column("model_version_tags", "workspace")
    op.drop_column("registered_model_aliases", "workspace")
    op.drop_column("registered_model_tags", "workspace")
    op.drop_column("model_versions", "workspace")
    op.drop_column("registered_models", "workspace")
    op.drop_column("experiments", "workspace")
    op.drop_column("evaluation_datasets", "workspace")
    op.drop_column("webhooks", "workspace")
    op.drop_column("jobs", "workspace")

    op.create_primary_key("registered_model_pk", "registered_models", ["name"])
    op.create_primary_key("model_version_pk", "model_versions", ["name", "version"])
    op.create_primary_key("registered_model_tag_pk", "registered_model_tags", ["key", "name"])
    op.create_primary_key("model_version_tag_pk", "model_version_tags", ["key", "name", "version"])
    op.create_primary_key(
        "registered_model_alias_pk", "registered_model_aliases", ["name", "alias"]
    )

    op.create_foreign_key(
        "model_versions_name_fkey",
        "model_versions",
        "registered_models",
        ["name"],
        ["name"],
        onupdate="CASCADE",
    )
    op.create_foreign_key(
        "registered_model_tags_name_fkey",
        "registered_model_tags",
        "registered_models",
        ["name"],
        ["name"],
        onupdate="CASCADE",
    )
    op.create_foreign_key(
        "registered_model_alias_name_fkey",
        "registered_model_aliases",
        "registered_models",
        ["name"],
        ["name"],
        onupdate="CASCADE",
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "model_version_tags_mv_fkey",
        "model_version_tags",
        "model_versions",
        ["name", "version"],
        ["name", "version"],
        onupdate="CASCADE",
    )

    op.drop_constraint("uq_model_definitions_workspace_name", "model_definitions", type_="unique")
    op.drop_constraint("uq_endpoints_workspace_name", "endpoints", type_="unique")
    op.drop_constraint("uq_secrets_workspace_secret_name", "secrets", type_="unique")

    op.drop_column("model_definitions", "workspace")
    op.drop_column("endpoints", "workspace")
    op.drop_column("secrets", "workspace")

    op.create_index("unique_model_definition_name", "model_definitions", ["name"], unique=True)
    op.create_index("unique_endpoint_name", "endpoints", ["name"], unique=True)
    op.create_index("unique_secret_name", "secrets", ["secret_name"], unique=True)
    op.create_index(
        "index_jobs_name_status_creation_time",
        "jobs",
        ["job_name", "status", "creation_time"],
    )

    op.drop_table("workspaces")

    op.create_unique_constraint("uq_experiments_name", "experiments", ["name"])

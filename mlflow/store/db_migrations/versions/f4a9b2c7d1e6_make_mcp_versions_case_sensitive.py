"""make MCP server versions case-sensitive

Create Date: 2026-09-16

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql, mysql

from mlflow.exceptions import MlflowException

revision = "f4a9b2c7d1e6"
down_revision = "8b6ad64cb8f5"
branch_labels = None
depends_on = None


_VERSION_COLUMNS = (
    ("mcp_server_versions", "version", False),
    ("mcp_server_version_tags", "version", False),
    ("mcp_server_aliases", "version", False),
    ("mcp_access_endpoints", "server_version", True),
)


def _check_version_references(connection, collation):
    versions = sa.table(
        "mcp_server_versions", sa.column("workspace"), sa.column("name"), sa.column("version")
    )
    for table_name, name_column, version_column in (
        ("mcp_server_version_tags", "name", "version"),
        ("mcp_server_aliases", "name", "version"),
        ("mcp_access_endpoints", "server_name", "server_version"),
    ):
        references = sa.table(
            table_name, sa.column("workspace"), sa.column(name_column), sa.column(version_column)
        )
        reference_version = references.c[version_column]
        # Join using the existing collation, then compare using the migration's new collation.
        row = connection.execute(
            sa
            .select(
                references.c.workspace,
                references.c[name_column],
                reference_version,
                versions.c.version,
            )
            .select_from(
                references.join(
                    versions,
                    sa.and_(
                        references.c.workspace == versions.c.workspace,
                        references.c[name_column] == versions.c.name,
                        reference_version == versions.c.version,
                    ),
                )
            )
            .where(reference_version.collate(collation) != versions.c.version.collate(collation))
            .limit(1)
        ).first()
        if row is not None:
            workspace, name, reference, version = row
            raise MlflowException(
                "Cannot make MCP server versions case-sensitive: "
                f"{table_name}.{version_column} contains {reference!r} for server {name!r} "
                f"in workspace {workspace!r}, but the stored version is {version!r}. "
                "No schema or data changes have been made by this migration. "
                "Back up the database and correct MCP version references in "
                "mcp_server_version_tags, mcp_server_aliases, and mcp_access_endpoints "
                "to match the exact stored version text before retrying the upgrade."
            )


def _change_version_type(new_type):
    op.drop_constraint(
        "mcp_server_version_tags_version_fkey",
        "mcp_server_version_tags",
        type_="foreignkey",
    )
    op.drop_index("ix_mcp_access_endpoints_version", table_name="mcp_access_endpoints")
    op.drop_constraint("mcp_server_version_tags_pk", "mcp_server_version_tags", type_="primary")
    op.drop_constraint("mcp_server_versions_pk", "mcp_server_versions", type_="primary")

    for table_name, column_name, nullable in _VERSION_COLUMNS:
        op.alter_column(
            table_name,
            column_name,
            existing_type=sa.String(128),
            type_=new_type,
            existing_nullable=nullable,
        )

    op.create_primary_key(
        "mcp_server_versions_pk",
        "mcp_server_versions",
        ["workspace", "name", "version"],
    )
    op.create_primary_key(
        "mcp_server_version_tags_pk",
        "mcp_server_version_tags",
        ["workspace", "name", "version", "key"],
    )
    op.create_foreign_key(
        "mcp_server_version_tags_version_fkey",
        "mcp_server_version_tags",
        "mcp_server_versions",
        ["workspace", "name", "version"],
        ["workspace", "name", "version"],
        ondelete="CASCADE",
        onupdate="CASCADE",
    )
    op.create_index(
        "ix_mcp_access_endpoints_version",
        "mcp_access_endpoints",
        ["workspace", "server_name", "server_version"],
    )


def upgrade():
    connection = op.get_bind()
    dialect = connection.dialect.name
    if dialect == "mysql":
        _check_version_references(connection, "utf8mb4_bin")
        _change_version_type(mysql.VARCHAR(128, collation="utf8mb4_bin"))
    elif dialect == "mssql":
        _check_version_references(connection, "SQL_Latin1_General_CP1_CS_AS")
        _change_version_type(mssql.VARCHAR(128, collation="SQL_Latin1_General_CP1_CS_AS"))


def downgrade():
    dialect = op.get_bind().dialect.name
    if dialect in ("mysql", "mssql"):
        _change_version_type(sa.String(128))

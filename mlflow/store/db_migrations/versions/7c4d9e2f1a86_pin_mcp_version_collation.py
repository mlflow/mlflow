"""pin case-sensitive collation on MCP version columns

SemVer (https://semver.org/#spec-item-11) compares prerelease and build
metadata identifiers as ASCII, so "1.0.0-A" and "1.0.0-a" are distinct
versions. MySQL and SQL Server default string columns to a case-insensitive
collation, which collapses those two values into a single row and lets
mcp_server_versions.get()/create() silently mix up versions. Pin a
case-sensitive collation on every column that stores an MCP version string.

Create Date: 2026-09-16 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql, mysql

# revision identifiers, used by Alembic.
revision = "7c4d9e2f1a86"
down_revision = "b7e2c1a4d9f3"
branch_labels = None
depends_on = None

_FK_NAME = "mcp_server_version_tags_version_fkey"
_VERSIONS_PK = "mcp_server_versions_pk"
_TAGS_PK = "mcp_server_version_tags_pk"
_ACCESS_ENDPOINTS_VERSION_IX = "ix_mcp_access_endpoints_version"


def _version_type():
    return (
        sa
        .String(128)
        .with_variant(mysql.VARCHAR(128, collation="utf8mb4_bin"), "mysql")
        .with_variant(mssql.VARCHAR(128, collation="SQL_Latin1_General_CP1_CS_AS"), "mssql")
    )


def upgrade():
    dialect = op.get_bind().dialect.name
    version_type = _version_type()
    # MySQL (error 3780) and SQL Server both refuse a foreign key whose two
    # sides disagree on collation, so the child FK must be dropped before
    # either side of it is altered, then recreated once both sides match.
    needs_fk_cycle = dialect in ("mysql", "mssql")
    # SQL Server additionally refuses to ALTER COLUMN a column that backs an
    # active PRIMARY KEY constraint (error 5074). `version` is part of the
    # primary key on both mcp_server_versions and mcp_server_version_tags,
    # so those two constraints must be dropped and recreated around the
    # ALTER as well. MySQL and SQLite/PostgreSQL don't need this.
    needs_pk_cycle = dialect == "mssql"

    if needs_fk_cycle:
        with op.batch_alter_table("mcp_server_version_tags") as batch_op:
            batch_op.drop_constraint(_FK_NAME, type_="foreignkey")

    with op.batch_alter_table("mcp_server_versions") as batch_op:
        if needs_pk_cycle:
            batch_op.drop_constraint(_VERSIONS_PK, type_="primary")
        batch_op.alter_column(
            "version", existing_type=sa.String(128), type_=version_type, existing_nullable=False
        )
        if needs_pk_cycle:
            batch_op.create_primary_key(_VERSIONS_PK, ["workspace", "name", "version"])

    with op.batch_alter_table("mcp_server_version_tags") as batch_op:
        if needs_pk_cycle:
            batch_op.drop_constraint(_TAGS_PK, type_="primary")
        batch_op.alter_column(
            "version", existing_type=sa.String(128), type_=version_type, existing_nullable=False
        )
        if needs_pk_cycle:
            batch_op.create_primary_key(_TAGS_PK, ["workspace", "name", "version", "key"])
        if needs_fk_cycle:
            batch_op.create_foreign_key(
                _FK_NAME,
                "mcp_server_versions",
                ["workspace", "name", "version"],
                ["workspace", "name", "version"],
                ondelete="CASCADE",
                onupdate="CASCADE",
            )

    with op.batch_alter_table("mcp_server_aliases") as batch_op:
        batch_op.alter_column(
            "version", existing_type=sa.String(128), type_=version_type, existing_nullable=False
        )

    with op.batch_alter_table("mcp_access_endpoints") as batch_op:
        # SQL Server also refuses to ALTER COLUMN a column backing an index
        # (error 5074); ix_mcp_access_endpoints_version covers server_version.
        if needs_pk_cycle:
            batch_op.drop_index(_ACCESS_ENDPOINTS_VERSION_IX)
        batch_op.alter_column(
            "server_version",
            existing_type=sa.String(128),
            type_=version_type,
            existing_nullable=True,
        )
        if needs_pk_cycle:
            batch_op.create_index(
                _ACCESS_ENDPOINTS_VERSION_IX, ["workspace", "server_name", "server_version"]
            )


def downgrade():
    dialect = op.get_bind().dialect.name
    needs_fk_cycle = dialect in ("mysql", "mssql")
    needs_pk_cycle = dialect == "mssql"

    with op.batch_alter_table("mcp_access_endpoints") as batch_op:
        if needs_pk_cycle:
            batch_op.drop_index(_ACCESS_ENDPOINTS_VERSION_IX)
        batch_op.alter_column(
            "server_version",
            existing_type=_version_type(),
            type_=sa.String(128),
            existing_nullable=True,
        )
        if needs_pk_cycle:
            batch_op.create_index(
                _ACCESS_ENDPOINTS_VERSION_IX, ["workspace", "server_name", "server_version"]
            )

    with op.batch_alter_table("mcp_server_aliases") as batch_op:
        batch_op.alter_column(
            "version", existing_type=_version_type(), type_=sa.String(128), existing_nullable=False
        )

    if needs_fk_cycle:
        with op.batch_alter_table("mcp_server_version_tags") as batch_op:
            batch_op.drop_constraint(_FK_NAME, type_="foreignkey")

    with op.batch_alter_table("mcp_server_version_tags") as batch_op:
        if needs_pk_cycle:
            batch_op.drop_constraint(_TAGS_PK, type_="primary")
        batch_op.alter_column(
            "version", existing_type=_version_type(), type_=sa.String(128), existing_nullable=False
        )
        if needs_pk_cycle:
            batch_op.create_primary_key(_TAGS_PK, ["workspace", "name", "version", "key"])

    with op.batch_alter_table("mcp_server_versions") as batch_op:
        if needs_pk_cycle:
            batch_op.drop_constraint(_VERSIONS_PK, type_="primary")
        batch_op.alter_column(
            "version", existing_type=_version_type(), type_=sa.String(128), existing_nullable=False
        )
        if needs_pk_cycle:
            batch_op.create_primary_key(_VERSIONS_PK, ["workspace", "name", "version"])

    if needs_fk_cycle:
        with op.batch_alter_table("mcp_server_version_tags") as batch_op:
            batch_op.create_foreign_key(
                _FK_NAME,
                "mcp_server_versions",
                ["workspace", "name", "version"],
                ["workspace", "name", "version"],
                ondelete="CASCADE",
                onupdate="CASCADE",
            )

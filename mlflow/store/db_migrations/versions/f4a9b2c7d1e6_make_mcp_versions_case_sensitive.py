"""make MCP server versions case-sensitive

Create Date: 2026-09-16

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql, mysql

revision = "f4a9b2c7d1e6"
down_revision = "b7e2c1a4d9f3"
branch_labels = None
depends_on = None


_VERSION_COLUMNS = (
    ("mcp_server_versions", "version", False),
    ("mcp_server_version_tags", "version", False),
    ("mcp_server_aliases", "version", False),
    ("mcp_access_endpoints", "server_version", True),
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
    dialect = op.get_bind().dialect.name
    if dialect == "mysql":
        _change_version_type(mysql.VARCHAR(128, collation="utf8mb4_bin"))
    elif dialect == "mssql":
        _change_version_type(mssql.VARCHAR(128, collation="SQL_Latin1_General_CP1_CS_AS"))


def downgrade():
    dialect = op.get_bind().dialect.name
    if dialect in ("mysql", "mssql"):
        _change_version_type(sa.String(128))

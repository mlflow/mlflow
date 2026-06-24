"""add MCP server registry tables

Create Date: 2026-05-11

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql

revision = "a8b9c0d1e2f3"
down_revision = "b7e4c1a90f23"
branch_labels = None
depends_on = None


def _get_json_type():
    dialect_name = op.get_bind().dialect.name
    if dialect_name == "mssql":
        return mssql.JSON
    else:
        return sa.JSON


def upgrade():
    json_type = _get_json_type()

    op.create_table(
        "mcp_servers",
        sa.Column(
            "workspace",
            sa.String(length=63),
            nullable=False,
            server_default=sa.text("'default'"),
        ),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("display_name", sa.String(length=256), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("icons", json_type, nullable=True),
        sa.Column("created_by", sa.String(length=256), nullable=True),
        sa.Column("last_updated_by", sa.String(length=256), nullable=True),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column("last_updated_at", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("workspace", "name", name="mcp_servers_pk"),
    )

    op.create_table(
        "mcp_server_versions",
        sa.Column(
            "workspace",
            sa.String(length=63),
            nullable=False,
            server_default=sa.text("'default'"),
        ),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("version", sa.String(length=128), nullable=False),
        sa.Column("version_major", sa.Integer(), nullable=False),
        sa.Column("version_minor", sa.Integer(), nullable=False),
        sa.Column("version_patch", sa.Integer(), nullable=False),
        sa.Column("version_prerelease_sort_key", sa.String(length=512), nullable=True),
        sa.Column("server_json", json_type, nullable=False),
        sa.Column("display_name", sa.String(length=256), nullable=True),
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'draft'"),
        ),
        sa.Column("tools", json_type, nullable=True),
        sa.Column("source", sa.String(length=512), nullable=True),
        sa.Column("created_by", sa.String(length=256), nullable=True),
        sa.Column("last_updated_by", sa.String(length=256), nullable=True),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column("last_updated_at", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["workspace", "name"],
            ["mcp_servers.workspace", "mcp_servers.name"],
            ondelete="CASCADE",
            onupdate="CASCADE",
            name="mcp_server_versions_server_fkey",
        ),
        sa.PrimaryKeyConstraint("workspace", "name", "version", name="mcp_server_versions_pk"),
    )

    op.create_table(
        "mcp_server_tags",
        sa.Column(
            "workspace",
            sa.String(length=63),
            nullable=False,
            server_default=sa.text("'default'"),
        ),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("key", sa.String(length=250), nullable=False),
        sa.Column("value", sa.String(length=5000), nullable=True),
        sa.ForeignKeyConstraint(
            ["workspace", "name"],
            ["mcp_servers.workspace", "mcp_servers.name"],
            ondelete="CASCADE",
            onupdate="CASCADE",
            name="mcp_server_tags_server_fkey",
        ),
        sa.PrimaryKeyConstraint("workspace", "name", "key", name="mcp_server_tags_pk"),
    )

    op.create_table(
        "mcp_server_version_tags",
        sa.Column(
            "workspace",
            sa.String(length=63),
            nullable=False,
            server_default=sa.text("'default'"),
        ),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("version", sa.String(length=128), nullable=False),
        sa.Column("key", sa.String(length=250), nullable=False),
        sa.Column("value", sa.String(length=5000), nullable=True),
        sa.ForeignKeyConstraint(
            ["workspace", "name", "version"],
            [
                "mcp_server_versions.workspace",
                "mcp_server_versions.name",
                "mcp_server_versions.version",
            ],
            ondelete="CASCADE",
            onupdate="CASCADE",
            name="mcp_server_version_tags_version_fkey",
        ),
        sa.PrimaryKeyConstraint(
            "workspace",
            "name",
            "version",
            "key",
            name="mcp_server_version_tags_pk",
        ),
    )

    op.create_table(
        "mcp_server_aliases",
        sa.Column(
            "workspace",
            sa.String(length=63),
            nullable=False,
            server_default=sa.text("'default'"),
        ),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("alias", sa.String(length=256), nullable=False),
        sa.Column("version", sa.String(length=128), nullable=False),
        sa.ForeignKeyConstraint(
            ["workspace", "name"],
            ["mcp_servers.workspace", "mcp_servers.name"],
            ondelete="CASCADE",
            onupdate="CASCADE",
            name="mcp_server_aliases_server_fkey",
        ),
        sa.PrimaryKeyConstraint("workspace", "name", "alias", name="mcp_server_aliases_pk"),
    )

    op.create_table(
        "mcp_access_bindings",
        sa.Column("binding_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "workspace",
            sa.String(length=63),
            nullable=False,
            server_default=sa.text("'default'"),
        ),
        sa.Column("server_name", sa.String(length=256), nullable=False),
        sa.Column("server_version", sa.String(length=128), nullable=True),
        sa.Column("server_alias", sa.String(length=256), nullable=True),
        sa.Column("endpoint_url", sa.String(length=2048), nullable=False),
        sa.Column(
            "transport_type",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'streamable-http'"),
        ),
        sa.Column("created_by", sa.String(length=256), nullable=True),
        sa.Column("last_updated_by", sa.String(length=256), nullable=True),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column("last_updated_at", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["workspace", "server_name"],
            ["mcp_servers.workspace", "mcp_servers.name"],
            ondelete="CASCADE",
            onupdate="CASCADE",
            name="mcp_access_bindings_server_fkey",
        ),
        sa.PrimaryKeyConstraint("binding_id", name="mcp_access_bindings_pk"),
    )

    # Keep this support index narrow enough for MySQL's 3072-byte key limit.
    # Latest resolution still orders by prerelease sort key and raw version in
    # SQL; they are just not part of the index because the coarse candidate
    # pruning and major/minor/patch ordering are the important indexed portion.
    op.create_index(
        "idx_mcp_server_versions_latest",
        "mcp_server_versions",
        [
            "workspace",
            "name",
            "status",
            sa.text("version_major DESC"),
            sa.text("version_minor DESC"),
            sa.text("version_patch DESC"),
        ],
    )

    op.create_index(
        "ix_mcp_access_bindings_server_name",
        "mcp_access_bindings",
        ["workspace", "server_name"],
    )
    op.create_index(
        "ix_mcp_access_bindings_version",
        "mcp_access_bindings",
        ["workspace", "server_name", "server_version"],
    )
    op.create_index(
        "ix_mcp_access_bindings_alias",
        "mcp_access_bindings",
        ["workspace", "server_name", "server_alias"],
    )


def downgrade():
    op.drop_index("ix_mcp_access_bindings_alias", table_name="mcp_access_bindings")
    op.drop_index("ix_mcp_access_bindings_version", table_name="mcp_access_bindings")
    op.drop_index("ix_mcp_access_bindings_server_name", table_name="mcp_access_bindings")
    op.drop_index("idx_mcp_server_versions_latest", table_name="mcp_server_versions")
    op.drop_table("mcp_access_bindings")
    op.drop_table("mcp_server_aliases")
    op.drop_table("mcp_server_version_tags")
    op.drop_table("mcp_server_tags")
    op.drop_table("mcp_server_versions")
    op.drop_table("mcp_servers")

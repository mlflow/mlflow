"""add span MCP server links

Create Date: 2026-10-26

"""

import sqlalchemy as sa
from alembic import op

revision = "c8f3a1b9d2e7"
down_revision = "f4a9b2c7d1e6"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "span_mcp_server_links",
        sa.Column("trace_id", sa.String(50), nullable=False),
        sa.Column("span_id", sa.String(50), nullable=False),
        sa.Column("mcp_workspace", sa.String(63), nullable=False),
        sa.Column("mcp_name", sa.String(256), nullable=False),
        sa.Column("mcp_version", sa.String(256), nullable=True),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint(
            "trace_id", "span_id", "mcp_workspace", "mcp_name", name="span_mcp_server_links_pk"
        ),
        sa.ForeignKeyConstraint(
            ["trace_id"],
            ["trace_info.request_id"],
            name="fk_span_mcp_server_links_trace_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_span_mcp_server_links_version",
        "span_mcp_server_links",
        ["mcp_workspace", "mcp_name", "mcp_version", "trace_id"],
    )


def downgrade():
    op.drop_index("ix_span_mcp_server_links_version", table_name="span_mcp_server_links")
    op.drop_table("span_mcp_server_links")

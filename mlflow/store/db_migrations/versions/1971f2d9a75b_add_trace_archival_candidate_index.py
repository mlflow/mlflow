"""add trace archival candidate index

Revision ID: 1971f2d9a75b
Revises: 6f8d9c3b2a1e

Create Date: 2026-08-09 00:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "1971f2d9a75b"
down_revision = "6f8d9c3b2a1e"
branch_labels = None
depends_on = None


def upgrade():
    op.create_index(
        "index_trace_info_timestamp_ms_request_id",
        "trace_info",
        ["timestamp_ms", "request_id"],
    )


def downgrade():
    op.drop_index("index_trace_info_timestamp_ms_request_id", table_name="trace_info")

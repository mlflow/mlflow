"""add trace_archival_location column to workspaces table

Per-workspace override for trace repository root. When set, used instead of
server global --trace-archival-location for archival and retrieval in that workspace.

Create Date: 2026-03-04 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "b8c9d0e1f2a3"
down_revision = "76601a5f987d"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("workspaces", schema=None) as batch_op:
        batch_op.add_column(sa.Column("trace_archival_location", sa.Text(), nullable=True))


def downgrade():
    with op.batch_alter_table("workspaces", schema=None) as batch_op:
        batch_op.drop_column("trace_archival_location")

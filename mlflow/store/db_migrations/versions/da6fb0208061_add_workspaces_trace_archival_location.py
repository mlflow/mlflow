"""add trace archival workspace columns

Revision ID: da6fb0208061
Revises: 7d34483879f0

Create Date: 2026-04-03 10:50:46.501372

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "da6fb0208061"
down_revision = "7d34483879f0"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("workspaces", schema=None) as batch_op:
        batch_op.add_column(sa.Column("trace_archival_location", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column("trace_archival_retention", sa.String(length=32), nullable=True)
        )


def downgrade():
    with op.batch_alter_table("workspaces", schema=None) as batch_op:
        batch_op.drop_column("trace_archival_retention")
        batch_op.drop_column("trace_archival_location")

"""add exclude_content to endpoints table

Create Date: 2026-07-12 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "e7f8a9b0c1d2"
down_revision = "b7e4c1a90f23"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("endpoints", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("exclude_content", sa.Boolean(), nullable=False, server_default="0")
        )


def downgrade():
    with op.batch_alter_table("endpoints", schema=None) as batch_op:
        batch_op.drop_column("exclude_content")

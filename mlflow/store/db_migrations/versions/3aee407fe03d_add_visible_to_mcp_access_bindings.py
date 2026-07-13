"""add visible column to mcp_access_bindings

Create Date: 2026-07-10 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "3aee407fe03d"
down_revision = "a8b9c0d1e2f3"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("mcp_access_bindings") as batch_op:
        batch_op.add_column(
            sa.Column("visible", sa.Boolean(), nullable=False, server_default=sa.text("1"))
        )


def downgrade():
    with op.batch_alter_table("mcp_access_bindings") as batch_op:
        batch_op.drop_column("visible")

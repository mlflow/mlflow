"""add hidden_connect_options column to mcp_server_versions

Create Date: 2026-07-10 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "b4c5d6e7f8a9"
down_revision = "3aee407fe03d"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("mcp_server_versions") as batch_op:
        batch_op.add_column(sa.Column("hidden_connect_options", sa.JSON(), nullable=True))


def downgrade():
    with op.batch_alter_table("mcp_server_versions") as batch_op:
        batch_op.drop_column("hidden_connect_options")

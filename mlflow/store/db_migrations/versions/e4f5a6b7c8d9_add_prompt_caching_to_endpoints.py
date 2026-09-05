"""add prompt_caching to endpoints table"""

import sqlalchemy as sa
from alembic import op

revision = "e4f5a6b7c8d9"
down_revision = "b7e2c1a4d9f3"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("endpoints", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("prompt_caching", sa.Boolean(), nullable=False, server_default="0")
        )



def downgrade():
    with op.batch_alter_table("endpoints", schema=None) as batch_op:
        batch_op.drop_column("prompt_caching")

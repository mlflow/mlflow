"""add routing strategy to endpoints

Create Date: 2025-12-18 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "c9d4e5f6a7b8"
down_revision = "5d2d30f0abce"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("endpoints", schema=None) as batch_op:
        batch_op.add_column(sa.Column("routing_strategy", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("fallback_config_json", sa.Text(), nullable=True))


def downgrade():
    with op.batch_alter_table("endpoints", schema=None) as batch_op:
        batch_op.drop_column("fallback_config_json")
        batch_op.drop_column("routing_strategy")

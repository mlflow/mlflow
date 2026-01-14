"""add experiment_id to endpoints table

Create Date: 2025-01-13 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "d0e1f2a3b4c5"
down_revision = "2c33131f4dae"
branch_labels = None
depends_on = None


def upgrade():
    # Add experiment_id column to endpoints table
    with op.batch_alter_table("endpoints", schema=None) as batch_op:
        batch_op.add_column(sa.Column("experiment_id", sa.String(length=32), nullable=True))


def downgrade():
    # Remove experiment_id column from endpoints table
    with op.batch_alter_table("endpoints", schema=None) as batch_op:
        batch_op.drop_column("experiment_id")

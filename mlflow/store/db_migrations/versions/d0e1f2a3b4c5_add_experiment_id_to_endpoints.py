"""add experiment_id and usage_tracking to endpoints table

Create Date: 2025-01-13 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "d0e1f2a3b4c5"
down_revision = "d3e4f5a6b7c8"
branch_labels = None
depends_on = None


def upgrade():
    # Add experiment_id and usage_tracking columns to endpoints table
    with op.batch_alter_table("endpoints", schema=None) as batch_op:
        batch_op.add_column(sa.Column("experiment_id", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column("usage_tracking", sa.Boolean(), nullable=False, server_default="0")
        )
        batch_op.create_foreign_key(
            "fk_endpoints_experiment_id",
            "experiments",
            ["experiment_id"],
            ["experiment_id"],
            ondelete="SET NULL",
        )


def downgrade():
    # Remove experiment_id and usage_tracking columns from endpoints table
    with op.batch_alter_table("endpoints", schema=None) as batch_op:
        batch_op.drop_constraint("fk_endpoints_experiment_id", type_="foreignkey")
        batch_op.drop_column("usage_tracking")
        batch_op.drop_column("experiment_id")

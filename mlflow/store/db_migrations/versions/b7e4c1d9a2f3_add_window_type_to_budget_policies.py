"""add window_type to budget_policies

Adds a ``window_type`` column to the ``budget_policies`` table so a budget policy can
measure spend over a sliding window (the trailing duration ending now) instead of fixed,
calendar-aligned windows. Existing policies keep fixed windows.

Create Date: 2026-11-23 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "b7e4c1d9a2f3"
down_revision = "f4a9b2c7d1e6"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("budget_policies", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "window_type",
                sa.String(length=32),
                nullable=False,
                server_default=sa.text("'FIXED'"),
            )
        )


def downgrade():
    with op.batch_alter_table("budget_policies", schema=None) as batch_op:
        # SQL Server refuses to drop a column that still has a DEFAULT constraint.
        batch_op.drop_column("window_type", mssql_drop_default=True)

"""add calls_per_minute to endpoints

Create Date: 2026-09-22 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "c3d9f1a75b28"
down_revision = "b7e2c1a4d9f3"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("endpoints", sa.Column("calls_per_minute", sa.Integer(), nullable=True))


def downgrade():
    op.drop_column("endpoints", "calls_per_minute")

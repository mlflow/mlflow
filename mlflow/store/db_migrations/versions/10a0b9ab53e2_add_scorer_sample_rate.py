"""Add sample_rate column to scorer_versions table

This migration adds a sample_rate column to the scorer_versions table to support
the update_scorer API. The sample_rate field allows configuration of scorer
sampling rates (0.0 to 1.0).

The default value of 0.0 ensures backward compatibility with existing scorers.

Create Date: 2025-01-29 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "10a0b9ab53e2"
down_revision = "bf29a5ff90ea"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("scorer_versions", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("sample_rate", sa.Float, nullable=False, server_default="0.0")
        )


def downgrade():
    with op.batch_alter_table("scorer_versions", schema=None) as batch_op:
        batch_op.drop_column("sample_rate")

"""Add sample_rate and sampling_strategy columns to scorer_versions table

This migration adds sample_rate and sampling_strategy columns to the scorer_versions table
to support the update_registered_scorer_sampling API. These fields allow configuration
of scorer sampling behavior:
- sample_rate: The fraction of traces to score (0.0 to 1.0)
- sampling_strategy: Optional strategy for selecting which traces to score

The default values (0.0 for sample_rate, NULL for sampling_strategy) ensure backward
compatibility with existing scorers.

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
        batch_op.add_column(sa.Column("sampling_strategy", sa.Text, nullable=True))


def downgrade():
    with op.batch_alter_table("scorer_versions", schema=None) as batch_op:
        batch_op.drop_column("sample_rate")
        batch_op.drop_column("sampling_strategy")

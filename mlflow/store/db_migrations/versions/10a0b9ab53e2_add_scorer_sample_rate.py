"""Add sample_rate, filter_string, and sampling_strategy columns to scorer_versions table

This migration adds scheduling and sampling configuration columns to the scorer_versions table
to support the update_registered_scorer_sampling API. These fields allow configuration
of scorer sampling behavior:
- sample_rate: The fraction of traces to score (0.0 to 1.0)
- filter_string: Optional filter string for selecting which traces to score
- sampling_strategy: Coordination mode for multiple versions
  ("independent"/"shared"/"partitioned")

The default values (0.0 for sample_rate, NULL for filter_string, 0 for sampling_strategy)
ensure backward compatibility with existing scorers.

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
        batch_op.add_column(sa.Column("filter_string", sa.Text, nullable=True))
        batch_op.add_column(
            sa.Column("sampling_strategy", sa.Integer, nullable=False, server_default="0")
        )


def downgrade():
    with op.batch_alter_table("scorer_versions", schema=None) as batch_op:
        batch_op.drop_column("sample_rate")
        batch_op.drop_column("filter_string")
        batch_op.drop_column("sampling_strategy")

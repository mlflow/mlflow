"""add sample_rate to scorer_versions table

Create Date: 2025-09-24 01:13:22.408104

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "10a0b9ab53e2"
down_revision = "bf29a5ff90ea"
branch_labels = None
depends_on = None


def upgrade():
    # Add sample_rate column to scorer_versions table
    with op.batch_alter_table("scorer_versions", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("sample_rate", sa.Float(precision=53), nullable=False, server_default="0.0")
        )


def downgrade():
    # Remove sample_rate column from scorer_versions table
    with op.batch_alter_table("scorer_versions", schema=None) as batch_op:
        batch_op.drop_column("sample_rate")

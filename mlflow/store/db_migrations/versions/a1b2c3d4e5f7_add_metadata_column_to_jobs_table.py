"""add job_metadata column to jobs table

Create Date: 2026-03-18 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f7"
down_revision = "76601a5f987d"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("jobs", sa.Column("job_metadata", sa.JSON(), nullable=True))


def downgrade():
    with op.batch_alter_table("jobs") as batch_op:
        batch_op.drop_column("job_metadata")

"""add creator column to jobs

Records the authenticated user who created a job so the basic-auth plugin can
enforce per-job ownership. Nullable: jobs created without
authentication (or before this migration) have no creator.

Create Date: 2026-08-13 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "b7e2c1a4d9f3"
down_revision = "6f8d9c3b2a1e"
branch_labels = None
depends_on = None


def upgrade():
    # Batch mode so the ALTER runs on SQLite as well as server DBs.
    with op.batch_alter_table("jobs") as batch_op:
        batch_op.add_column(sa.Column("creator", sa.String(255), nullable=True))


def downgrade():
    with op.batch_alter_table("jobs") as batch_op:
        batch_op.drop_column("creator")

"""add allowlisted_models to secrets

Create Date: 2026-07-23 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "a1c2e3f40b56"
down_revision = "b7e4c1a90f23"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("secrets", schema=None) as batch_op:
        batch_op.add_column(sa.Column("allowlisted_models", sa.Text(), nullable=True))


def downgrade():
    with op.batch_alter_table("secrets", schema=None) as batch_op:
        batch_op.drop_column("allowlisted_models")

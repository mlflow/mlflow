"""add display_name to endpoint_bindings

Create Date: 2026-01-21 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "d3e4f5a6b7c8"
down_revision = "2c33131f4dae"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("endpoint_bindings", schema=None) as batch_op:
        batch_op.add_column(sa.Column("display_name", sa.String(length=255), nullable=True))


def downgrade():
    with op.batch_alter_table("endpoint_bindings", schema=None) as batch_op:
        batch_op.drop_column("display_name")

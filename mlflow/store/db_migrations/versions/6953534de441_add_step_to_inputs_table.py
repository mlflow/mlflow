"""add step to inputs table

Revision ID: 6953534de441
Revises: 400f98739977
Create Date: 2025-02-13 11:50:07.098121

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "6953534de441"
down_revision = "400f98739977"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("inputs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("step", sa.BigInteger(), nullable=False, server_default="0"))


def downgrade():
    pass

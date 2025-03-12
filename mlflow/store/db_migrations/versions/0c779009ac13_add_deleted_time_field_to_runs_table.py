"""add deleted_time field to runs table

Revision ID: 0c779009ac13
Revises: bd07f7e963c5
Create Date: 2022-07-27 14:13:36.162861

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0c779009ac13"
down_revision = "bd07f7e963c5"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("runs", sa.Column("deleted_time", sa.BigInteger, nullable=True, default=None))


def downgrade():
    pass

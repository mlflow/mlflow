"""Add creation_time and last_update_time to experiments table

Create Date: 2022-08-26 21:16:59.164858

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "97727af70f4d"
down_revision = "cc1f77228345"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("experiments", sa.Column("creation_time", sa.BigInteger(), nullable=True))
    op.add_column("experiments", sa.Column("last_update_time", sa.BigInteger(), nullable=True))


def downgrade():
    pass

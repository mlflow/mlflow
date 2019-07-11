"""allow nulls for metric values

Revision ID: 181f10493468
Revises: 90e64c465722
Create Date: 2019-07-10 22:40:18.787993

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '181f10493468'
down_revision = '90e64c465722'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('metrics', sa.Column('is_nan', sa.Boolean(), nullable=True, server_default="0"))


def downgrade():
    pass

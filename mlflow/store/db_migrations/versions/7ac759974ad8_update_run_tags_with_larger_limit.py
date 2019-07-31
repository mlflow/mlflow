"""Update run tags with larger limit

Revision ID: 7ac759974ad8
Revises: df50e92ffc5e
Create Date: 2019-07-30 16:36:54.256382

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '7ac759974ad8'
down_revision = 'df50e92ffc5e'
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column("tags", 'value',
                    existing_type=sa.String(250),
                    type_=sa.String(5000),
                    existing_nullable=True,
                    existing_server_default=None)


def downgrade():
    pass

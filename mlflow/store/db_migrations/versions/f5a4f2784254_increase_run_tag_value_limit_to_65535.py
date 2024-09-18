"""increase run tag value limit to 65535

Revision ID: f5a4f2784254
Revises: 4465047574b1
Create Date: 2024-09-18 08:53:51.552934

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f5a4f2784254'
down_revision = '4465047574b1'
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass

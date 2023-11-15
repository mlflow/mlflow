"""extend digest max size to 44

Revision ID: da24556b8093
Revises: acf3f17fdcc7
Create Date: 2023-11-15 20:54:25.104933

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "da24556b8093"
down_revision = "acf3f17fdcc7"
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass

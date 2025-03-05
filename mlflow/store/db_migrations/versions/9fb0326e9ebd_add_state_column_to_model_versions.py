"""increase run tag value limit to 8000

Revision ID: 9fb0326e9ebd
Revises: 4465047574b1
Create Date: 2025-03-03 

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "9fb0326e9ebd"
down_revision = "4465047574b1"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'model_versions',
        sa.Column('state', sa.String(length=10), nullable=False, server_default='New')
    )


def downgrade():
    op.drop_column('model_versions', 'state')

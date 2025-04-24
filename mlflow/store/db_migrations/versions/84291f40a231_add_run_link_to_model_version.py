"""add run_link to model_version

Revision ID: 84291f40a231
Revises: 27a6a02d2cf1
Create Date: 2020-07-16 13:45:56.178092

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "84291f40a231"
down_revision = "27a6a02d2cf1"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "model_versions", sa.Column("run_link", sa.String(500), nullable=True, default=None)
    )


def downgrade():
    pass

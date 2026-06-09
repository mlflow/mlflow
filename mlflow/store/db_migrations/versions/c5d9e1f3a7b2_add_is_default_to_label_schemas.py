"""add is_default to label_schemas

Revision ID: c5d9e1f3a7b2
Revises: b7e4c1a90f23

Create Date: 2026-06-09 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "c5d9e1f3a7b2"
down_revision = "b7e4c1a90f23"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "label_schemas",
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade():
    op.drop_column("label_schemas", "is_default")

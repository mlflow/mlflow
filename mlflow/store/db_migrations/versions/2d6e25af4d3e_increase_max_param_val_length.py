"""increase max param val length from 500 to 8000

Revision ID: 2d6e25af4d3e
Revises: 7f2a7d5fae7d
Create Date: 2023-09-25 13:59:04.231744

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "2d6e25af4d3e"
down_revision = "7f2a7d5fae7d"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("params") as batch_op:
        batch_op.alter_column(
            "value",
            existing_type=sa.String(500),
            # We choose 8000 because it's the minimum max_length for
            # a VARCHAR column in all supported database types.
            type_=sa.String(8000),
            existing_nullable=False,
            existing_server_default=None,
        )


def downgrade():
    pass

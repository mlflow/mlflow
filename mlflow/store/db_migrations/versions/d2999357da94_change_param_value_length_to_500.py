"""change param value length to 500

Revision ID: d2999357da94
Revises: bd07f7e963c5
Create Date: 2022-07-27 22:00:13.187596

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "d2999357da94"
down_revision = "bd07f7e963c5"
branch_labels = None
depends_on = None


def upgrade():
    """
    Enlarge the maximum param value length to 500.
    """
    with op.batch_alter_table("params") as batch_op:
        batch_op.alter_column(
            "value",
            existing_type=sa.String(250),
            type_=sa.String(500),
            existing_nullable=False,
            nullable=False,
        )


def downgrade():
    pass

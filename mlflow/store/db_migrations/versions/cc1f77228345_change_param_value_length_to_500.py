"""change param value length to 500

Revision ID: cc1f77228345
Revises: 0c779009ac13
Create Date: 2022-08-04 22:40:56.960003

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "cc1f77228345"
down_revision = "0c779009ac13"
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

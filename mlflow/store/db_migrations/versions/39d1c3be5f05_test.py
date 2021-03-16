"""test

Revision ID: 39d1c3be5f05
Revises: a8c4a736bde6
Create Date: 2021-03-16 20:40:24.214667

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "39d1c3be5f05"
down_revision = "a8c4a736bde6"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("latest_metrics") as batch_op:
        batch_op.alter_column(
            "is_nan", type_=sa.types.Boolean(create_constraint=True), nullable=False, default=False
        )


def downgrade():
    pass

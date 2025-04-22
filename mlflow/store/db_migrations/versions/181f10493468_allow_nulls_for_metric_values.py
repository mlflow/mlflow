"""allow nulls for metric values

Revision ID: 181f10493468
Revises: 90e64c465722
Create Date: 2019-07-10 22:40:18.787993

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "181f10493468"
down_revision = "90e64c465722"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("metrics") as batch_op:
        batch_op.alter_column("value", type_=sa.types.Float(precision=53), nullable=False)
        batch_op.add_column(
            sa.Column(
                "is_nan", sa.Boolean(create_constraint=False), nullable=False, server_default="0"
            )
        )
        batch_op.drop_constraint(constraint_name="metric_pk", type_="primary")
        batch_op.create_primary_key(
            constraint_name="metric_pk",
            columns=["key", "timestamp", "step", "run_uuid", "value", "is_nan"],
        )


def downgrade():
    pass

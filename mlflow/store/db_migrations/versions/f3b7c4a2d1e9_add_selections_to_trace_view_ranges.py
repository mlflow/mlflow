"""add input_selections and output_selections to trace_view_ranges

Create Date: 2026-04-09 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "f3b7c4a2d1e9"
down_revision = "eb885a9619f6"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("trace_view_ranges", schema=None) as batch_op:
        batch_op.add_column(sa.Column("input_selections", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("output_selections", sa.Text(), nullable=True))


def downgrade():
    with op.batch_alter_table("trace_view_ranges", schema=None) as batch_op:
        batch_op.drop_column("output_selections")
        batch_op.drop_column("input_selections")

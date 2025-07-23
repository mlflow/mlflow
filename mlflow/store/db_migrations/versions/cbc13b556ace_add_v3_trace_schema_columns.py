"""add V3 trace schema columns

Revision ID: cbc13b556ace
Revises: bda7b8c39065
Create Date: 2025-06-17 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "cbc13b556ace"
down_revision = "bda7b8c39065"
branch_labels = None
depends_on = None


def upgrade():
    # Add V3 specific columns to trace_info table
    with op.batch_alter_table("trace_info", schema=None) as batch_op:
        batch_op.add_column(sa.Column("client_request_id", sa.String(length=50), nullable=True))
        batch_op.add_column(sa.Column("request_preview", sa.String(length=1000), nullable=True))
        batch_op.add_column(sa.Column("response_preview", sa.String(length=1000), nullable=True))


def downgrade():
    with op.batch_alter_table("trace_info", schema=None) as batch_op:
        batch_op.drop_column("response_preview")
        batch_op.drop_column("request_preview")
        batch_op.drop_column("client_request_id")

"""add V3 trace schema columns

Revision ID: cbc13b556ace
Revises: 5b0e9adcef9c
Create Date: 2025-06-17 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "cbc13b556ace"
down_revision = "6953534de441"
branch_labels = None
depends_on = None


def upgrade():
    # Add V3 specific columns to trace_info table
    with op.batch_alter_table("trace_info", schema=None) as batch_op:
        batch_op.add_column(sa.Column("client_request_id", sa.String(length=50), nullable=True))
        batch_op.add_column(sa.Column("request_preview", sa.String(length=8000), nullable=True))
        batch_op.add_column(sa.Column("response_preview", sa.String(length=8000), nullable=True))


def downgrade():
    with op.batch_alter_table("trace_info", schema=None) as batch_op:
        batch_op.drop_column("response_preview")
        batch_op.drop_column("request_preview")
        batch_op.drop_column("client_request_id")

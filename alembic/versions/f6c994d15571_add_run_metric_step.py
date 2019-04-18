"""add run metric step

Revision ID: f6c994d15571
Revises: 
Create Date: 2019-04-16 21:58:37.356633

"""
from alembic import op
from sqlalchemy import Column, BigInteger


# revision identifiers, used by Alembic.
revision = 'f6c994d15571'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(table_name='metrics', column=Column('step', BigInteger, default=0))


def downgrade():
    op.drop_column('metrics', 'step')

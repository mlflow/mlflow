"""create periodic_jobs table

Revision ID: b0ec69213791
Revises: 
Create Date: 2019-09-02 13:37:43.652186

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b0ec69213791'
down_revision = '7ac759974ad8'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('periodic_jobs',
                    sa.Column('job_name', sa.String(length=256), primary_key=True, nullable=False),
                    sa.Column('last_execution', sa.BIGINT),
                    sa.PrimaryKeyConstraint('job_name', name='periodic_jobs_pk')
                    )


def downgrade():
    op.drop_table('periodic_jobs')

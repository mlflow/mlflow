"""add env_vars column to job table

Create Date: 2025-09-29 11:15:27.552595

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '4209a452f04b'
down_revision = 'bf29a5ff90ea'
branch_labels = None
depends_on = None


def upgrade():
    # Add env_vars column to jobs table
    op.add_column(
        "jobs",
        sa.Column("env_vars", sa.Text(), nullable=True),
    )


def downgrade():
    # Remove env_vars column from jobs table
    op.drop_column("jobs", "env_vars")

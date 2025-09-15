"""add jobs table

Create Date: 2025-09-11 17:39:31.569736

"""
import time
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'bf29a5ff90ea'
down_revision = '71994744cf8e'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("creation_time", sa.BigInteger(), default=lambda: int(time.time() * 1000)),
        sa.Column("function", sa.String(length=500), nullable=False),
        sa.Column("params", sa.Text(), nullable=False),
        sa.Column("status", sa.Integer(), nullable=False),
        sa.Column("result", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer, default=0),
        sa.PrimaryKeyConstraint("id", name="jobs_pk"),
    )
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.create_index(
            "index_jobs_function_status_creation_time",
            ["function", "status", "creation_time"],
            unique=False,
        )


def downgrade():
    op.drop_table("jobs")

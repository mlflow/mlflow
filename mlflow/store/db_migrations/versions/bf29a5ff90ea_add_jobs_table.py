"""add jobs table

Create Date: 2025-09-11 17:39:31.569736

"""

import time

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "bf29a5ff90ea"
down_revision = "3da73c924c2f"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column(
            "creation_time",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.Column("function_fullname", sa.String(length=500), nullable=False),
        sa.Column("params", sa.Text(), nullable=False),
        sa.Column("timeout", sa.Float(precision=53), nullable=True),
        sa.Column("status", sa.Integer(), nullable=False),
        sa.Column("result", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), default=0, nullable=False),
        sa.Column(
            "last_update_time",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="jobs_pk"),
    )
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.create_index(
            "index_jobs_function_status_creation_time",
            ["function_fullname", "status", "creation_time"],
            unique=False,
        )


def downgrade():
    op.drop_table("jobs")

"""add job executor persistence model

Create Date: 2026-04-13 14:19:33.487151

"""

import time

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql

# revision identifiers, used by Alembic.
revision = "dc11669786a5"
down_revision = "7d34483879f0"
branch_labels = None
depends_on = None

_PENDING = 0
_RUNNING = 1
_CANCELED = 5


def _get_json_type():
    dialect_name = op.get_bind().dialect.name
    if dialect_name == "mssql":
        return mssql.JSON
    return sa.JSON


def _cancel_non_terminal_jobs():
    jobs = sa.table("jobs", sa.column("status"), sa.column("last_update_time"))
    op.execute(
        jobs
        .update()
        .where(jobs.c.status.in_([_PENDING, _RUNNING]))
        .values(status=_CANCELED, last_update_time=int(time.time() * 1000))
    )


def upgrade():
    json_type = _get_json_type()

    op.create_table(
        "scheduler_leases",
        sa.Column("lease_key", sa.String(length=255), nullable=False),
        sa.Column("acquired_at", sa.BigInteger(), nullable=False),
        sa.Column("ttl_seconds", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("lease_key", name="scheduler_leases_pk"),
    )

    op.create_table(
        "job_locks",
        sa.Column("lock_key", sa.String(length=255), nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("acquired_at", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["jobs.id"],
            name="fk_job_locks_job_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("lock_key", name="job_locks_pk"),
    )

    with op.batch_alter_table("job_locks") as batch_op:
        batch_op.create_index("index_job_locks_job_id", ["job_id"], unique=False)

    with op.batch_alter_table("jobs") as batch_op:
        batch_op.add_column(sa.Column("executor_backend", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("lease_expires_at", sa.BigInteger(), nullable=True))
        batch_op.add_column(sa.Column("status_message", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("progress_payload", json_type, nullable=True))
        batch_op.add_column(sa.Column("progress_updated_at", sa.BigInteger(), nullable=True))
        batch_op.add_column(sa.Column("token_hash", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("scoped_permissions", json_type, nullable=True))
        batch_op.create_index(
            "index_jobs_status_lease_expires_at",
            ["status", "lease_expires_at"],
            unique=False,
        )

    _cancel_non_terminal_jobs()


def downgrade():
    with op.batch_alter_table("jobs") as batch_op:
        batch_op.drop_index("index_jobs_status_lease_expires_at")
        batch_op.drop_column("scoped_permissions")
        batch_op.drop_column("token_hash")
        batch_op.drop_column("progress_updated_at")
        batch_op.drop_column("progress_payload")
        batch_op.drop_column("status_message")
        batch_op.drop_column("lease_expires_at")
        batch_op.drop_column("executor_backend")

    with op.batch_alter_table("job_locks") as batch_op:
        batch_op.drop_index("index_job_locks_job_id")

    op.drop_table("job_locks")
    op.drop_table("scheduler_leases")

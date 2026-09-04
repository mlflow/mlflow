"""add span experiment and start time index

Revision ID: c1d2e3f4a5b6
Revises: a8b9c0d1e2f3

Create Date: 2026-07-10 00:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "c1d2e3f4a5b6"
down_revision = "a8b9c0d1e2f3"
branch_labels = None
depends_on = None

# The composite index's leftmost experiment_id column also supports experiment-only filters, so
# it replaces the existing experiment_id index rather than being added alongside it.


def upgrade():
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table("spans", schema=None) as batch_op:
            batch_op.drop_index("index_spans_experiment_id")
            batch_op.create_index(
                "index_spans_experiment_id_start_time",
                ["experiment_id", "start_time_unix_nano"],
                unique=False,
            )
    else:
        op.drop_index("index_spans_experiment_id", table_name="spans")
        op.create_index(
            "index_spans_experiment_id_start_time",
            "spans",
            ["experiment_id", "start_time_unix_nano"],
            unique=False,
        )


def downgrade():
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table("spans", schema=None) as batch_op:
            batch_op.drop_index("index_spans_experiment_id_start_time")
            batch_op.create_index("index_spans_experiment_id", ["experiment_id"], unique=False)
    else:
        op.drop_index("index_spans_experiment_id_start_time", table_name="spans")
        op.create_index(
            "index_spans_experiment_id",
            "spans",
            ["experiment_id"],
            unique=False,
        )

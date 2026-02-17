"""Add composite index on metrics(run_uuid, key, step)

This index dramatically speeds up the get-history-bulk-interval endpoint
by allowing SELECT DISTINCT step queries to use an index-only scan instead
of scanning all metric rows. On tables with 75M+ rows, this reduces the
endpoint response time from ~2.5s to ~100ms.

Create Date: 2026-02-17 16:00:30.898522

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "138e9eb975d4"
down_revision = "1b5f0d9ad7c1"
branch_labels = None
depends_on = None


def upgrade():
    op.create_index(
        "index_metrics_run_uuid_key_step",
        "metrics",
        ["run_uuid", "key", "step"],
    )


def downgrade():
    op.drop_index("index_metrics_run_uuid_key_step", table_name="metrics")

"""add composite index on metrics (run_uuid, key, step)

Create Date: 2026-03-10 20:46:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "a5b4c3d2e1f0"
down_revision = "c3d6457b6d8a"
branch_labels = None
depends_on = None


def upgrade():
    # Add composite index to speed up metric history queries that filter by
    # run_uuid and key, and order by step. See https://github.com/mlflow/mlflow/issues/12813
    op.create_index("index_metrics_run_uuid_key_step", "metrics", ["run_uuid", "key", "step"])


def downgrade():
    op.drop_index("index_metrics_run_uuid_key_step", table_name="metrics")

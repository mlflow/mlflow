"""add metric step

Revision ID: 451aebb31d03
Revises:
Create Date: 2019-04-22 15:29:24.921354

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "451aebb31d03"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("metrics", sa.Column("step", sa.BigInteger(), nullable=False, server_default="0"))
    # Use batch mode so that we can run "ALTER TABLE" statements against SQLite
    # databases (see more info at https://alembic.sqlalchemy.org/en/latest/
    # batch.html#running-batch-migrations-for-sqlite-and-other-databases)
    with op.batch_alter_table("metrics") as batch_op:
        batch_op.drop_constraint(constraint_name="metric_pk", type_="primary")
        batch_op.create_primary_key(
            constraint_name="metric_pk", columns=["key", "timestamp", "step", "run_uuid", "value"]
        )


def downgrade():
    # This migration cannot safely be downgraded; once metric data with the same
    # (key, timestamp, run_uuid, value) are inserted (differing only in their `step`), we cannot
    # revert to a schema where (key, timestamp, run_uuid, value) is the metric primary key.
    pass

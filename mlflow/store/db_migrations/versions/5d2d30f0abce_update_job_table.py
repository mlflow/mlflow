"""update job table

Create Date: 2025-12-16 16:31:47.921120

"""

from alembic import op
from sqlalchemy import String

# revision identifiers, used by Alembic.
revision = "5d2d30f0abce"
down_revision = "b7c8d9e0f1a2"
branch_labels = None
depends_on = None


def upgrade():
    # Rename column `function_fullname` -> `job_name` and update the related index
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        # Drop old index that referenced `function_fullname`
        batch_op.drop_index("index_jobs_function_status_creation_time")
        # Rename the column
        batch_op.alter_column(
            "function_fullname", new_column_name="job_name", existing_type=String(500)
        )

    with op.batch_alter_table("jobs", schema=None) as batch_op:
        # Recreate the index referencing the new column name
        batch_op.create_index(
            "index_jobs_name_status_creation_time",
            ["job_name", "status", "creation_time"],
            unique=False,
        )


def downgrade():
    # Revert column rename `job_name` -> `function_fullname` and restore the original index
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.drop_index("index_jobs_name_status_creation_time")
        batch_op.alter_column(
            "job_name", new_column_name="function_fullname", existing_type=String(500)
        )

    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.create_index(
            "index_jobs_function_status_creation_time",
            ["function_fullname", "status", "creation_time"],
            unique=False,
        )

"""add_is_nan_constraint_for_metrics_tables_if_necessary

Create Date: 2021-03-16 20:40:24.214667

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "39d1c3be5f05"
down_revision = "a8c4a736bde6"
branch_labels = None
depends_on = None


def upgrade():
    # This part of the migration is only relevant for users who installed sqlalchemy 1.4.0 with
    # MLflow <= 1.14.1. In sqlalchemy 1.4.0, the default value of `create_constraint` for
    # `sqlalchemy.Boolean` was changed to `False` from `True`:
    # https://github.com/sqlalchemy/sqlalchemy/blob/e769ba4b00859ac8c95610ed149da4d940eac9d0/lib/sqlalchemy/sql/sqltypes.py#L1841
    # To ensure that a check constraint is always present on the `is_nan` column in the
    # `latest_metrics` table, we perform an `alter_column` and explicitly set `create_constraint`
    # to `True`
    with op.batch_alter_table("latest_metrics") as batch_op:
        batch_op.alter_column(
            "is_nan", type_=sa.types.Boolean(create_constraint=True), nullable=False
        )

    # Introduce a check constraint on the `is_nan` column from the `metrics` table, which was
    # missing prior to this migration
    with op.batch_alter_table("metrics") as batch_op:
        batch_op.alter_column(
            "is_nan", type_=sa.types.Boolean(create_constraint=True), nullable=False
        )


def downgrade():
    pass

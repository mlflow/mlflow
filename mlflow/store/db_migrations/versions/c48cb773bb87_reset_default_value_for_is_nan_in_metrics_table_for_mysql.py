"""reset_default_value_for_is_nan_in_metrics_table_for_mysql

Create Date: 2021-04-02 15:43:28.466043

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "c48cb773bb87"
down_revision = "39d1c3be5f05"
branch_labels = None
depends_on = None


def upgrade():
    # This part of the migration is only relevant for MySQL.
    # In 39d1c3be5f05_add_is_nan_constraint_for_metrics_tables_if_necessary.py
    # (added in MLflow 1.15.0), `alter_column` is called on the `is_nan` column in the `metrics`
    # table without specifying `existing_server_default`. This alters the column default value to
    # NULL in MySQL (see the doc below).
    #
    # https://alembic.sqlalchemy.org/en/latest/ops.html#alembic.operations.Operations.alter_column
    #
    # To revert this change, set the default column value to "0" by specifying `server_default`
    bind = op.get_bind()
    if bind.engine.name == "mysql":
        with op.batch_alter_table("metrics") as batch_op:
            batch_op.alter_column(
                "is_nan",
                type_=sa.types.Boolean(create_constraint=True),
                nullable=False,
                server_default="0",
            )


def downgrade():
    pass

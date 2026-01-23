"""add cascade delete to trace and span experiment foreign keys

Create Date: 2026-01-23 14:50:00.000000

"""

import sqlalchemy as sa
from alembic import op

from mlflow.exceptions import MlflowException
from mlflow.store.tracking.dbmodels.models import (
    SqlExperiment,
    SqlOnlineScoringConfig,
    SqlSpan,
    SqlTraceInfo,
)

# revision identifiers, used by Alembic.
revision = "0f1a2b3c4d5e"
down_revision = "84291f40a231"
branch_labels = None
depends_on = None


def get_foreign_key_name(table_name, column_name, referred_table_name):
    """Helper function to get the foreign key constraint name."""
    conn = op.get_bind()
    metadata = sa.MetaData()
    metadata.bind = conn
    table = sa.Table(table_name, metadata, autoload_with=conn)

    for constraint in table.foreign_key_constraints:
        if (
            constraint.referred_table.name == referred_table_name
            and column_name in constraint.column_keys
        ):
            return constraint.name

    raise MlflowException(
        f"Unable to find the foreign key constraint name from {table_name}.{column_name} "
        f"to {referred_table_name}. "
        f"All foreign key constraints in {table_name} table: \n{table.foreign_key_constraints}"
    )


def upgrade():
    dialect_name = op.get_context().dialect.name

    # List of tables and their foreign keys to experiments.experiment_id that need CASCADE
    tables_to_update = [
        (SqlTraceInfo.__tablename__, "experiment_id", "fk_trace_info_experiment_id"),
        (SqlSpan.__tablename__, "experiment_id", "fk_spans_experiment_id"),
        (
            SqlOnlineScoringConfig.__tablename__,
            "experiment_id",
            "fk_online_scoring_configs_experiment_id",
        ),
    ]

    for table_name, column_name, new_fk_name in tables_to_update:
        if dialect_name == "sqlite":
            # SQLite requires batch operations to alter foreign keys
            # See https://alembic.sqlalchemy.org/en/latest/batch.html
            with op.batch_alter_table(
                table_name,
                schema=None,
                naming_convention={
                    "fk": "fk_%(table_name)s_%(column_0_name)s",
                },
            ) as batch_op:
                # Drop the old foreign key constraint
                try:
                    old_fk_name = f"fk_{table_name}_{column_name}"
                    batch_op.drop_constraint(old_fk_name, type_="foreignkey")
                except Exception:
                    # If the constraint name doesn't match the pattern, try the provided name
                    batch_op.drop_constraint(new_fk_name, type_="foreignkey")

                # Create the new foreign key constraint with CASCADE
                batch_op.create_foreign_key(
                    new_fk_name,
                    SqlExperiment.__tablename__,
                    [column_name],
                    ["experiment_id"],
                    ondelete="CASCADE",
                )
        else:
            # For other databases (MySQL, PostgreSQL, etc.)
            try:
                old_fk_name = get_foreign_key_name(
                    table_name, column_name, SqlExperiment.__tablename__
                )
                op.drop_constraint(old_fk_name, table_name, type_="foreignkey")
            except Exception:
                # If we can't find the constraint, try the standardized name
                try:
                    op.drop_constraint(new_fk_name, table_name, type_="foreignkey")
                except Exception:
                    # Skip if constraint doesn't exist (defensive programming)
                    pass

            # Create the new foreign key constraint with CASCADE
            op.create_foreign_key(
                new_fk_name,
                table_name,
                SqlExperiment.__tablename__,
                [column_name],
                ["experiment_id"],
                ondelete="CASCADE",
            )


def downgrade():
    pass

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
down_revision = "d3e4f5a6b7c8"
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

    # Update trace_info FK
    table_name = SqlTraceInfo.__tablename__
    column_name = "experiment_id"
    new_fk_name = "fk_trace_info_experiment_id"

    if dialect_name == "sqlite":
        with op.batch_alter_table(
            table_name,
            schema=None,
            naming_convention={
                "fk": "fk_%(table_name)s_%(column_0_name)s",
            },
        ) as batch_op:
            try:
                old_fk_name = f"fk_{table_name}_{column_name}"
                batch_op.drop_constraint(old_fk_name, type_="foreignkey")
            except Exception:
                batch_op.drop_constraint(new_fk_name, type_="foreignkey")

            batch_op.create_foreign_key(
                new_fk_name,
                SqlExperiment.__tablename__,
                [column_name],
                ["experiment_id"],
                ondelete="CASCADE",
            )
    else:
        try:
            old_fk_name = get_foreign_key_name(table_name, column_name, SqlExperiment.__tablename__)
            op.drop_constraint(old_fk_name, table_name, type_="foreignkey")
        except Exception:
            try:
                op.drop_constraint(new_fk_name, table_name, type_="foreignkey")
            except Exception:
                pass

        op.create_foreign_key(
            new_fk_name,
            table_name,
            SqlExperiment.__tablename__,
            [column_name],
            ["experiment_id"],
            ondelete="CASCADE",
        )

    # Update spans FK - special handling for SQLite due to computed column
    table_name = SqlSpan.__tablename__
    new_fk_name = "fk_spans_experiment_id"

    if dialect_name == "sqlite":
        # For SQLite, we need to use copy_from to exclude the computed column
        # We must include all existing constraints including the FKs
        with op.batch_alter_table(
            table_name,
            schema=None,
            naming_convention={
                "fk": "fk_%(table_name)s_%(column_0_name)s",
            },
            copy_from=sa.Table(
                table_name,
                sa.MetaData(),
                sa.Column("trace_id", sa.String(50), nullable=False),
                sa.Column("experiment_id", sa.Integer(), nullable=False),
                sa.Column("span_id", sa.String(50), nullable=False),
                sa.Column("parent_span_id", sa.String(50), nullable=True),
                sa.Column("name", sa.Text(), nullable=True),
                sa.Column("type", sa.String(500), nullable=True),
                sa.Column("status", sa.String(50), nullable=False),
                sa.Column("start_time_unix_nano", sa.BigInteger(), nullable=False),
                sa.Column("end_time_unix_nano", sa.BigInteger(), nullable=True),
                # Don't include duration_ns here - it's a computed column
                sa.Column("content", sa.Text(), nullable=False),
                sa.ForeignKeyConstraint(
                    ["trace_id"],
                    ["trace_info.request_id"],
                    name="fk_spans_trace_id",
                    ondelete="CASCADE",
                ),
                sa.ForeignKeyConstraint(
                    ["experiment_id"], ["experiments.experiment_id"], name="fk_spans_experiment_id"
                ),
            ),
        ) as batch_op:
            # Drop the old FK constraint and recreate with CASCADE
            batch_op.drop_constraint("fk_spans_experiment_id", type_="foreignkey")

            batch_op.create_foreign_key(
                new_fk_name,
                SqlExperiment.__tablename__,
                [column_name],
                ["experiment_id"],
                ondelete="CASCADE",
            )
    else:
        try:
            old_fk_name = get_foreign_key_name(table_name, column_name, SqlExperiment.__tablename__)
            op.drop_constraint(old_fk_name, table_name, type_="foreignkey")
        except Exception:
            try:
                op.drop_constraint(new_fk_name, table_name, type_="foreignkey")
            except Exception:
                pass

        op.create_foreign_key(
            new_fk_name,
            table_name,
            SqlExperiment.__tablename__,
            [column_name],
            ["experiment_id"],
            ondelete="CASCADE",
        )

    # Update online_scoring_configs FK
    table_name = SqlOnlineScoringConfig.__tablename__
    new_fk_name = "fk_online_scoring_configs_experiment_id"

    if dialect_name == "sqlite":
        with op.batch_alter_table(
            table_name,
            schema=None,
            naming_convention={
                "fk": "fk_%(table_name)s_%(column_0_name)s",
            },
        ) as batch_op:
            try:
                old_fk_name = f"fk_{table_name}_{column_name}"
                batch_op.drop_constraint(old_fk_name, type_="foreignkey")
            except Exception:
                batch_op.drop_constraint(new_fk_name, type_="foreignkey")

            batch_op.create_foreign_key(
                new_fk_name,
                SqlExperiment.__tablename__,
                [column_name],
                ["experiment_id"],
                ondelete="CASCADE",
            )
    else:
        try:
            old_fk_name = get_foreign_key_name(table_name, column_name, SqlExperiment.__tablename__)
            op.drop_constraint(old_fk_name, table_name, type_="foreignkey")
        except Exception:
            try:
                op.drop_constraint(new_fk_name, table_name, type_="foreignkey")
            except Exception:
                pass

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

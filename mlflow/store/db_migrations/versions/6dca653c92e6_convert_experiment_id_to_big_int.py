"""convert experiment_id from auto-increment to fixed length

Revision ID: 6dca653c92e6
Revises: 97727af70f4d
Create Date: 2022-10-06 15:47:21.228666

This migration converts the primary key `experiment_id` within the Experiments table from an
auto-incrementing primary key to a non-nullable unique-constrained Integer column. This is to
support concurrent experiment creation and avoid collisions.
"""
from alembic import op
import sqlalchemy as sa
import logging

from sqlalchemy import PrimaryKeyConstraint, ForeignKeyConstraint, Sequence
from sqlalchemy.inspection import inspect

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

# revision identifiers, used by Alembic.
revision = "6dca653c92e6"
down_revision = "97727af70f4d"
branch_labels = None
depends_on = None


def upgrade():
    # As part of MLflow 2.0 upgrade, the Experiment table's primary key `experiment_id`
    # has changed from an auto-incrementing column to a non-nullable unique-constrained Integer
    # column to support the uuid-based random id generation change.

    engine = op.get_bind()
    engine_name = engine.engine.name

    # NB: sqlite doesn't support foreign keys even if they are defined. Altering a constraint
    # in sqlite outside of batch operations doesn't work.
    if engine_name != "sqlite":

        foreign_keys_in_experiment_tags = inspect(engine).get_foreign_keys("experiment_tags")
        fk = foreign_keys_in_experiment_tags[0]
        op.drop_constraint(fk["name"], table_name="experiment_tags", type_="foreignkey")

        # NB: MSSQL and MySQL have special restrictions on batch updates that foreign keys.
        # In order to handle type casting modifications, these constraints need to be dropped
        # prior to any ALTER commands within the batch context. After type changes are complete, we
        # will recreate these foreign keys (and give them names so that inspection isn't
        # required in the future).

        foreign_keys_in_runs = inspect(engine).get_foreign_keys("runs")
        fk_run = foreign_keys_in_runs[0]
        op.drop_constraint(fk_run["name"], table_name="runs", type_="foreignkey")

        # NB: MSSQL requires that modifications to primary key columns do not have a primary key
        # status assigned to the columns. These primary keys will be recreated after altering
        # the columns typing.
        if engine_name == "mssql":
            op.drop_constraint("experiment_pk", table_name="experiments", type_="primary")
            op.drop_constraint("experiment_tag_pk", table_name="experiment_tags", type_="primary")

        experiments_table_args = PrimaryKeyConstraint("experiment_id", name="experiment_pk")
    else:
        experiments_table_args = []

    with op.batch_alter_table(
        "experiments",
        table_args=experiments_table_args,
    ) as batch_op:

        if engine_name == "mssql":
            batch_op.alter_column(
                "experiment_id",
                existing_type=sa.Integer,
                type_=sa.BigInteger,
                existing_nullable=False,
                nullable=False,
                existing_autoincrement=True,
                autoincrement=False,
                existing_server_default=None,
                existing_comment=None,
            )
        else:
            experiment_id_seq = Sequence("experiment_id_seq", start=1)
            batch_op.alter_column(
                "experiment_id",
                existing_type=sa.Integer,
                type_=sa.BigInteger,
                existing_nullable=False,
                nullable=False,
                existing_autoincrement=True,
                autoincrement=False,
                existing_server_default=experiment_id_seq.next_value(),
                server_default=None,
            )

    if engine_name == "sqlite":
        # NB: sqlite will perform an in-place copy of a table and recreate constraints as defined
        # in the `table_args` argument to the batch constructor.
        experiments_tags_table_args = (
            PrimaryKeyConstraint("key", "experiment_id", name="experiment_tag_pk"),
            ForeignKeyConstraint(
                columns=["experiment_id"], refcolumns=["experiments.experiment_id"]
            ),
        )
    else:
        # For postgres and mysql, the primary key definition will be applied to the altered table
        # if defined in the `table_args` argument (and will be ignored in mssql).
        experiments_tags_table_args = (
            PrimaryKeyConstraint("key", "experiment_id", name="experiment_tag_pk"),
        )

    with op.batch_alter_table(
        "experiment_tags",
        table_args=experiments_tags_table_args,
    ) as batch_op:
        batch_op.alter_column(
            "experiment_id",
            existing_type=sa.Integer,
            type_=sa.BigInteger,
            existing_nullable=False,
            nullable=False,
        )

    with op.batch_alter_table(
        "runs",
    ) as batch_op:
        batch_op.alter_column(
            "experiment_id",
            existing_type=sa.Integer,
            type_=sa.BigInteger,
            existing_nullable=True,
            nullable=True,
        )
        if engine_name == "sqlite":
            # sqlite will not port over unnamed constraints. Existing version has this
            # constraint defined as `CHECK()` rather than `CONSTRAINT <namme> CHECK()`
            batch_op.create_check_constraint(
                constraint_name="status",
                condition="status IN ('SCHEDULED', 'FAILED', 'FINISHED', 'RUNNING', 'KILLED')",
            )

    # NB: MSSQL identity columns cannot be modified. Copying data to new column.
    # Then, drooping original and renaming the new column.
    if engine_name == "mssql":

        with op.batch_alter_table("experiments") as batch_op:

            # create the new column
            batch_op.add_column(
                sa.Column("exp_id", sa.BigInteger, nullable=False), insert_before="experiment_id"
            )

            # perform data migration
            batch_op.execute("UPDATE experiments SET exp_id = experiment_id")

            # drop column
            batch_op.drop_column("experiment_id")

            # rename column
            batch_op.alter_column(column_name="exp_id", new_column_name="experiment_id")

    if engine_name != "sqlite":

        # NB: mssql requires that foreign keys reference primary keys prior to
        # creation of a foreign key. Recreate the primary keys that were previously dropped.
        if engine.engine.name == "mssql":
            op.create_primary_key(
                constraint_name="experiment_pk", table_name="experiments", columns=["experiment_id"]
            )
            op.create_primary_key(
                constraint_name="experiment_tag_pk",
                table_name="experiment_tags",
                columns=["key", "experiment_id"],
            )

        # Recreate the foreign key and name it for future direct reference
        op.create_foreign_key(
            constraint_name="fk_experiment_tag",
            source_table="experiment_tags",
            referent_table="experiments",
            local_cols=["experiment_id"],
            remote_cols=["experiment_id"],
        )

        op.create_foreign_key(
            constraint_name="fk_runs_experiment_id",
            source_table="runs",
            referent_table="experiments",
            local_cols=["experiment_id"],
            remote_cols=["experiment_id"],
        )

    _logger.info("Conversion of experiment_id from autoincrement complete!")


def downgrade():
    pass

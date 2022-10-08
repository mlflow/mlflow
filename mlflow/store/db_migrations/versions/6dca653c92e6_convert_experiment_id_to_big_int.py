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

from sqlalchemy import UniqueConstraint, PrimaryKeyConstraint

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

    bind = op.get_bind()

    if bind.engine.name != "sqlite":

        if bind.engine.name == "mssql":
            # One-time explicit naming for reference. When recreating this foreign key below,
            # we name it.
            fkey_constraint_experiment_tags = "FK__experimen__exper__4F7CD00D"
            op.drop_constraint(
                constraint_name=fkey_constraint_experiment_tags,
                table_name="experiment_tags",
                type_="foreignkey",
            )
        else:
            # Alembic unnamed constraint reflection works in MySQL and Postgres
            metadata = sa.MetaData(bind=bind)
            metadata.reflect()
            naming_convention = metadata.naming_convention

            with op.batch_alter_table(
                "experiment_tags", naming_convention=naming_convention
            ) as batch_op:
                batch_op.drop_constraint("experiments.experiment_id", type_="foreignkey")

        op.drop_constraint("experiment_tag_pk", table_name="experiment_tags", type_="primary")
        op.drop_constraint(
            constraint_name="experiment_pk", table_name="experiments", type_="primary"
        )
        op.alter_column(
            table_name="experiments",
            column_name="experiment_id",
            existing_type=sa.Integer,
            type_=sa.BigInteger,
            existing_nullable=False,
            nullable=False,
            autoincrement=False,
            existing_autoincrement=True,
        )
        op.alter_column(
            table_name="experiment_tags",
            column_name="experiment_id",
            existing_type=sa.Integer,
            type_=sa.BigInteger,
            existing_nullable=False,
            nullable=False,
        )
        op.alter_column(
            table_name="runs",
            column_name="experiment_id",
            existing_type=sa.Integer,
            type_=sa.BigInteger,
            existing_nullable=True,
            nullable=True,
        )

        op.create_unique_constraint(
            constraint_name="uq_experiment_id", table_name="experiments", columns=["experiment_id"]
        )
        op.create_primary_key(
            constraint_name="experiment_pk", table_name="experiments", columns=["experiment_id"]
        )
        op.create_foreign_key(
            constraint_name="fk_experiment_tag",
            source_table="experiment_tags",
            referent_table="experiments",
            local_cols=["experiment_id"],
            remote_cols=["experiment_id"],
            onupdate="CASCADE",
            ondelete="CASCADE",
        )
        op.create_primary_key(
            constraint_name="experiment_tag_pk",
            table_name="experiment_tags",
            columns=["key", "experiment_id"],
        )
    else:

        with op.batch_alter_table(
            "experiments",
            table_args=(
                UniqueConstraint("experiment_id"),
                PrimaryKeyConstraint("experiment_id", name="experiment_pk"),
            ),
        ) as batch_op:
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

        with op.batch_alter_table(
            "experiment_tags",
            table_args=(PrimaryKeyConstraint("key", "experiment_id", name="experiment_tag_pk"),),
        ) as batch_op:
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

        with op.batch_alter_table("runs") as batch_op:
            batch_op.alter_column(
                "experiment_id",
                existing_type=sa.Integer,
                type_=sa.BigInteger,
                existing_nullable=True,
                nullable=True,
                existing_autoincrement=False,
                autoincrement=False,
                existing_server_default=None,
                existing_comment=None,
            )

    _logger.info("Conversion of experiment_id from autoincrement complete!")


def downgrade():
    pass

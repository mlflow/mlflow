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

from sqlalchemy import PrimaryKeyConstraint
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
    # NB: sqlite doesn't support foreign keys even if they are defined. Altering a constraint
    # in sqlite outside of batch operations doesn't work.
    if engine.engine.name != "sqlite":

        foreign_keys_in_experiment_tags = inspect(engine).get_foreign_keys("experiment_tags")
        fk = foreign_keys_in_experiment_tags[0]
        op.drop_constraint(fk["name"], table_name="experiment_tags", type_="foreignkey")

    if engine.engine.name == "mssql":

        op.drop_constraint("experiment_pk", table_name="experiments", type_="primary")
        op.drop_constraint("experiment_tag_pk", table_name="experiment_tags", type_="primary")

    with op.batch_alter_table(
        "experiments",
        table_args=(PrimaryKeyConstraint("experiment_id", name="experiment_pk")),
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
    if engine.engine.name != "sqlite":
        # Recreate the foreign key and name it for future direct reference
        op.create_foreign_key(
            constraint_name="fk_experiment_tag",
            source_table="experiment_tags",
            referent_table="experiments",
            local_cols=["experiment_id"],
            remote_cols=["experiment_id"],
            onupdate="CASCADE",
            ondelete="CASCADE",
        )

        if engine.engine.name == "mmssql":
            op.create_primary_key(
                constraint_name="experiment_pk", table_name="experiments", columns=["experiment_id"]
            )
            op.create_primary_key(
                constraint_name="experiment_tag_pk",
                table_name="experiment_tags",
                columns=["key", "experiment_id"],
            )

    _logger.info("Conversion of experiment_id from autoincrement complete!")


def downgrade():
    pass

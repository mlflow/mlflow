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

from sqlalchemy import UniqueConstraint

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

    with op.batch_alter_table(
        "experiments", table_args=(UniqueConstraint("experiment_id"))
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
        "experiment_tags", table_args=(UniqueConstraint("experiment_id"))
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
            unique=True,
        )

    with op.batch_alter_table("runs", table_args=(UniqueConstraint("experiment_id"))) as batch_op:
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
            unique=True,
        )

    _logger.info("Conversion of experiment_id from autoincrement complete!")


def downgrade():
    pass

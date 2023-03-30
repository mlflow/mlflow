"""add inputs table

Revision ID: 30da9e9d8245
Revises: 7f2a7d5fae7d
Create Date: 2023-03-30 01:26:39.984889

"""
from alembic import op
import sqlalchemy as sa
from mlflow.store.tracking.dbmodels.models import SqlInputs


# revision identifiers, used by Alembic.
revision = "30da9e9d8245"
down_revision = "7f2a7d5fae7d"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        SqlInputs.__tablename__,
        sa.Column("input_uuid", sa.String(length=36), nullable=False),
        sa.Column("source_type", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column("source_id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column("destination_type", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column("destination_id", sa.String(length=36), primary_key=True, nullable=False),
        sa.PrimaryKeyConstraint(
            "source_type", "source_id", "destination_type", "destination_id", name="inputs_pk"
        ),
        sa.Index(f"index_{SqlInputs.__tablename__}_input_uuid", "input_uuid", unique=False),
        sa.Index(
            f"index_{SqlInputs.__tablename__}_destination_type_destination_id_source_type",
            "destination_type",
            "destination_id",
            "source_type",
            unique=False,
        ),
    )


def downgrade():
    pass

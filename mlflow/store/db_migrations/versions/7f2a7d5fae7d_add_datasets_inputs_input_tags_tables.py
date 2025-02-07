"""add datasets inputs input_tags tables

Revision ID: 7f2a7d5fae7d
Revises: 3500859a5d39
Create Date: 2023-03-23 09:48:27.775166

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mysql import MEDIUMTEXT

from mlflow.store.tracking.dbmodels.models import SqlDataset, SqlInput, SqlInputTag

# revision identifiers, used by Alembic.
revision = "7f2a7d5fae7d"
down_revision = "3500859a5d39"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        SqlDataset.__tablename__,
        sa.Column("dataset_uuid", sa.String(length=36), nullable=False),
        sa.Column(
            "experiment_id",
            sa.Integer(),
            sa.ForeignKey("experiments.experiment_id"),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("name", sa.String(length=500), primary_key=True, nullable=False),
        sa.Column("digest", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column("dataset_source_type", sa.String(length=36), nullable=False),
        sa.Column("dataset_source", sa.Text(), nullable=False),
        sa.Column("dataset_schema", sa.Text(), nullable=True),
        sa.Column("dataset_profile", sa.Text().with_variant(MEDIUMTEXT, "mysql"), nullable=True),
        sa.PrimaryKeyConstraint("experiment_id", "name", "digest", name="dataset_pk"),
        sa.Index(f"index_{SqlDataset.__tablename__}_dataset_uuid", "dataset_uuid", unique=False),
        sa.Index(
            f"index_{SqlDataset.__tablename__}_experiment_id_dataset_source_type",
            "experiment_id",
            "dataset_source_type",
            unique=False,
        ),
    )
    op.create_table(
        SqlInput.__tablename__,
        sa.Column("input_uuid", sa.String(length=36), nullable=False),
        sa.Column("source_type", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column("source_id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column("destination_type", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column("destination_id", sa.String(length=36), primary_key=True, nullable=False),
        sa.PrimaryKeyConstraint(
            "source_type", "source_id", "destination_type", "destination_id", name="inputs_pk"
        ),
        sa.Index(f"index_{SqlInput.__tablename__}_input_uuid", "input_uuid", unique=False),
        sa.Index(
            f"index_{SqlInput.__tablename__}_destination_type_destination_id_source_type",
            "destination_type",
            "destination_id",
            "source_type",
            unique=False,
        ),
    )
    op.create_table(
        SqlInputTag.__tablename__,
        sa.Column(
            "input_uuid",
            sa.String(length=36),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("name", sa.String(length=255), primary_key=True, nullable=False),
        sa.Column("value", sa.String(length=500), nullable=False),
        sa.PrimaryKeyConstraint("input_uuid", "name", name="input_tags_pk"),
    )


def downgrade():
    pass

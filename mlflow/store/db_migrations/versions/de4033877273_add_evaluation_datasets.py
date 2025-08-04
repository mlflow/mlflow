"""add evaluation datasets

Revision ID: de4033877273
Revises: 770bee3ae1dd
Create Date: 2025-07-28 13:05:53.982327

"""

import sqlalchemy as sa
from alembic import op

from mlflow.store.db.mutable_json import MutableJSON

# revision identifiers, used by Alembic.
revision = "de4033877273"
down_revision = "770bee3ae1dd"
branch_labels = None
depends_on = None


def upgrade():
    # Use the shared mutable JSON type that handles all backends
    json_type = MutableJSON

    # Create evaluation_datasets table
    op.create_table(
        "evaluation_datasets",
        sa.Column("dataset_id", sa.String(36), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("tags", json_type, nullable=True),
        sa.Column("schema", sa.Text(), nullable=True),
        sa.Column("profile", sa.Text(), nullable=True),
        sa.Column("digest", sa.String(64), nullable=True),
        sa.Column("created_time", sa.BigInteger(), nullable=True),
        sa.Column("last_update_time", sa.BigInteger(), nullable=True),
        sa.Column("created_by", sa.String(255), nullable=True),
        sa.Column("last_updated_by", sa.String(255), nullable=True),
        sa.PrimaryKeyConstraint("dataset_id", name="evaluation_datasets_pk"),
    )

    # Create indexes on evaluation_datasets
    with op.batch_alter_table("evaluation_datasets", schema=None) as batch_op:
        batch_op.create_index(
            "index_evaluation_datasets_name",
            ["name"],
            unique=False,
        )
        batch_op.create_index(
            "index_evaluation_datasets_created_time",
            ["created_time"],
            unique=False,
        )

    # Create evaluation_dataset_records table
    op.create_table(
        "evaluation_dataset_records",
        sa.Column("dataset_record_id", sa.String(36), nullable=False),
        sa.Column("dataset_id", sa.String(36), nullable=False),
        sa.Column("inputs", json_type, nullable=False),
        sa.Column("expectations", json_type, nullable=True),
        sa.Column("tags", json_type, nullable=True),
        sa.Column("source", json_type, nullable=True),
        sa.Column("source_id", sa.String(36), nullable=True),
        sa.Column("source_type", sa.String(255), nullable=True),
        sa.Column("created_time", sa.BigInteger(), nullable=True),
        sa.Column("last_update_time", sa.BigInteger(), nullable=True),
        sa.Column("created_by", sa.String(255), nullable=True),
        sa.Column("last_updated_by", sa.String(255), nullable=True),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.PrimaryKeyConstraint("dataset_record_id", name="evaluation_dataset_records_pk"),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["evaluation_datasets.dataset_id"],
            name="fk_evaluation_dataset_records_dataset_id",
            ondelete="CASCADE",
        ),
    )

    # Create indexes and unique constraint on evaluation_dataset_records
    with op.batch_alter_table("evaluation_dataset_records", schema=None) as batch_op:
        batch_op.create_index(
            "index_evaluation_dataset_records_dataset_id",
            ["dataset_id"],
            unique=False,
        )
        batch_op.create_unique_constraint(
            "unique_dataset_input",
            ["dataset_id", "input_hash"],
        )

    # Create entity_associations table
    op.create_table(
        "entity_associations",
        sa.Column("association_id", sa.String(36), nullable=False),
        sa.Column("source_type", sa.String(36), nullable=False),
        sa.Column("source_id", sa.String(36), nullable=False),
        sa.Column("destination_type", sa.String(36), nullable=False),
        sa.Column("destination_id", sa.String(36), nullable=False),
        sa.Column("created_time", sa.BigInteger(), nullable=True),
        sa.PrimaryKeyConstraint(
            "source_type",
            "source_id",
            "destination_type",
            "destination_id",
            name="entity_associations_pk",
        ),
    )

    # Create indexes on entity_associations
    with op.batch_alter_table("entity_associations", schema=None) as batch_op:
        batch_op.create_index(
            "index_entity_associations_association_id",
            ["association_id"],
            unique=False,
        )
        batch_op.create_index(
            "index_entity_associations_reverse_lookup",
            ["destination_type", "destination_id", "source_type", "source_id"],
            unique=False,
        )


def downgrade():
    # Drop tables in reverse order to respect foreign key constraints
    op.drop_table("entity_associations")
    op.drop_table("evaluation_dataset_records")
    op.drop_table("evaluation_datasets")

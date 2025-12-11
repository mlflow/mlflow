"""add evaluation datasets

Revision ID: 71994744cf8e
Revises: 534353b11cbc
Create Date: 2025-08-12 14:30:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql

# revision identifiers, used by Alembic.
revision = "71994744cf8e"
down_revision = "534353b11cbc"
branch_labels = None
depends_on = None


def _get_json_type():
    """Get appropriate JSON type for the current database."""
    dialect_name = op.get_bind().dialect.name
    if dialect_name == "mssql":
        # Use MSSQL-specific JSON type (stored as NVARCHAR(MAX))
        # This is available in SQLAlchemy 1.4+ and works with SQL Server 2016+
        return mssql.JSON
    else:
        # Use standard JSON type for other databases
        return sa.JSON


def upgrade():
    json_type = _get_json_type()

    # Create evaluation_datasets table
    op.create_table(
        "evaluation_datasets",
        sa.Column("dataset_id", sa.String(36), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        # Note: tags are stored in a separate table, not as JSON column
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

    # Create evaluation_dataset_tags table
    op.create_table(
        "evaluation_dataset_tags",
        sa.Column("dataset_id", sa.String(36), nullable=False),
        sa.Column("key", sa.String(255), nullable=False),
        sa.Column("value", sa.String(5000), nullable=True),
        sa.PrimaryKeyConstraint("dataset_id", "key", name="evaluation_dataset_tags_pk"),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["evaluation_datasets.dataset_id"],
            name="fk_evaluation_dataset_tags_dataset_id",
            ondelete="CASCADE",
        ),
    )

    # Create indexes on evaluation_dataset_tags
    with op.batch_alter_table("evaluation_dataset_tags", schema=None) as batch_op:
        batch_op.create_index(
            "index_evaluation_dataset_tags_dataset_id",
            ["dataset_id"],
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


def downgrade():
    # Drop tables in reverse order to respect foreign key constraints
    op.drop_table("evaluation_dataset_records")
    op.drop_table("evaluation_dataset_tags")
    op.drop_table("evaluation_datasets")

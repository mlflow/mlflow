"""add outputs to dataset record

Create Date: 2025-01-16 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql

# revision identifiers, used by Alembic.
revision = "3da73c924c2f"
down_revision = "71994744cf8e"
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
    """Add outputs column to evaluation_dataset_records table."""
    json_type = _get_json_type()

    # Add outputs column to evaluation_dataset_records table
    op.add_column("evaluation_dataset_records", sa.Column("outputs", json_type, nullable=True))


def downgrade():
    """Remove outputs column from evaluation_dataset_records table."""
    op.drop_column("evaluation_dataset_records", "outputs")

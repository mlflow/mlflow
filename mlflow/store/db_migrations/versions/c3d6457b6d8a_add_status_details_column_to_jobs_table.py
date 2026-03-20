"""add status_details column to jobs table

Create Date: 2026-03-20 09:48:33.248771

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql

# revision identifiers, used by Alembic.
revision = "c3d6457b6d8a"
down_revision = "76601a5f987d"
branch_labels = None
depends_on = None


def _get_json_type():
    """Get appropriate JSON type for the current database."""
    dialect_name = op.get_bind().dialect.name
    if dialect_name == "mssql":
        return mssql.JSON
    else:
        return sa.JSON


def upgrade():
    json_type = _get_json_type()
    op.add_column("jobs", sa.Column("status_details", json_type, nullable=True))


def downgrade():
    with op.batch_alter_table("jobs") as batch_op:
        batch_op.drop_column("status_details")

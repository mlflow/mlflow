"""add links column to spans table

Create Date: 2026-04-16 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql

# revision identifiers, used by Alembic.
revision = "a1d2e3f4b5c6"
down_revision = "7d34483879f0"
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
    with op.batch_alter_table("spans", schema=None) as batch_op:
        batch_op.add_column(sa.Column("links", json_type, nullable=True))


def downgrade():
    with op.batch_alter_table("spans", schema=None) as batch_op:
        batch_op.drop_column("links")

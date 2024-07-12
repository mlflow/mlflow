"""increase max dataset schema size

Revision ID: 4465047574b1
Revises: 5b0e9adcef9c
Create Date: 2024-07-09 12:54:33.775087

"""
import logging
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import MEDIUMTEXT

_logger = logging.getLogger(__name__)


# revision identifiers, used by Alembic.
revision = '4465047574b1'
down_revision = '5b0e9adcef9c'
branch_labels = None
depends_on = None


def upgrade():
    try:
        # For other database backends, the dataset_schema column already satisfies the new length
        if op.get_bind().engine.name == "mysql":
            op.alter_column("datasets", "dataset_schema", existing_type=sa.TEXT, type_=MEDIUMTEXT)
    except Exception as e:
        _logger.warning(
            "Failed to update dataset_schema column to MEDIUMTEXT type, it may not be supported "
            f"by your SQL database. Exception content: {e}"
        )


def downgrade():
    pass

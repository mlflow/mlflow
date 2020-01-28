"""drop_duplicate_killed_constraint

Revision ID: 0a8213491aaa
Revises: cfd24bdc0731
Create Date: 2020-01-28 15:26:14.757445

"""
import logging
from alembic import op

_logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision = '0a8213491aaa'
down_revision = 'cfd24bdc0731'
branch_labels = None
depends_on = None


def upgrade():
    # Attempt to drop any existing `status` constraints on the `runs` table. This operation
    # may fail against certain backends with different classes of Exception. For example,
    # in MySQL <= 8.0.15, dropping constraints produces an invalid `ALTER TABLE` expression.
    # Further, in certain versions of sqlite, `ALTER` (which is invoked by `drop_constraint`)
    # is unsupported on `CHECK` constraints. Accordingly, we catch the generic `Exception`
    # object because the failure modes are not well-enumerated or consistent across database
    # backends.
    try:
        op.drop_constraint(constraint_name="status", table_name="runs", type_="check")
    except Exception as e: # pylint: disable=broad-except
        _logger.warning(
            "Failed to drop check constraint. Dropping check constraints may not be supported"
            " by your SQL database. Exception content: %s", e)


def downgrade():
    pass

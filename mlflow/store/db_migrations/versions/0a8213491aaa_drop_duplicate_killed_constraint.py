"""drop_duplicate_killed_constraint

Create Date: 2020-01-28 15:26:14.757445

This migration drops a duplicate constraint on the `runs.status` column that was left as a byproduct
of an erroneous implementation of the `cfd24bdc0731_update_run_status_constraint_with_killed`
migration in MLflow 1.5. The implementation of this migration has since been fixed.
"""

import logging

from alembic import op

_logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision = "0a8213491aaa"
down_revision = "cfd24bdc0731"
branch_labels = None
depends_on = None


def upgrade():
    # Attempt to drop any existing `status` constraints on the `runs` table. This operation
    # may fail against certain backends with different classes of Exception. For example,
    # in MySQL <= 8.0.15, dropping constraints produces an invalid `ALTER TABLE` expression.
    # Further, in certain versions of sqlite, `ALTER` (which is invoked by `drop_constraint`)
    # is unsupported on `CHECK` constraints. Accordingly, we catch the generic `Exception`
    # object because the failure modes are not well-enumerated or consistent across database
    # backends. Because failures automatically stop batch operations and the `drop_constraint()`
    # operation is expected to fail under certain circumstances, we execute `drop_constraint()`
    # outside of the batch operation context.
    try:
        # For other database backends, the status check constraint is dropped by
        # cfd24bdc0731_update_run_status_constraint_with_killed.py
        if op.get_bind().engine.name == "mysql":
            op.drop_constraint(constraint_name="status", table_name="runs", type_="check")
    except Exception as e:
        _logger.warning(
            "Failed to drop check constraint. Dropping check constraints may not be supported"
            " by your SQL database. Exception content: %s",
            e,
        )


def downgrade():
    pass

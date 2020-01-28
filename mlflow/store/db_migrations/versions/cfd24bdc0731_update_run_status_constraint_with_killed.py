"""Update run status constraint with killed

Revision ID: cfd24bdc0731
Revises: 89d4b8295536
Create Date: 2019-10-11 15:55:10.853449

"""
import logging
from alembic import op
from mlflow.entities import RunStatus, ViewType
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.store.tracking.dbmodels.models import SqlRun, SourceTypes
from sqlalchemy import CheckConstraint, Enum, String

_logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision = 'cfd24bdc0731'
down_revision = '2b4d017a5e9b'
branch_labels = None
depends_on = None

new_run_statuses = [
    RunStatus.to_string(RunStatus.SCHEDULED),
    RunStatus.to_string(RunStatus.FAILED),
    RunStatus.to_string(RunStatus.FINISHED),
    RunStatus.to_string(RunStatus.RUNNING),
    RunStatus.to_string(RunStatus.KILLED)
]

previous_run_statuses = [
    RunStatus.to_string(RunStatus.SCHEDULED),
    RunStatus.to_string(RunStatus.FAILED),
    RunStatus.to_string(RunStatus.FINISHED),
    RunStatus.to_string(RunStatus.RUNNING)
]

# Certain SQL backends (e.g., SQLite) do not preserve CHECK constraints during migrations.
# For these backends, CHECK constraints must be specified as table arguments. Here, we define
# the collection of CHECK constraints that should be preserved when performing the migration.
# The "status" constraint is excluded from this set because it is explicitly modified
# within the migration's `upgrade()` routine.
check_constraint_table_args = [
    CheckConstraint(SqlRun.source_type.in_(SourceTypes), name='source_type'),
    CheckConstraint(SqlRun.lifecycle_stage.in_(LifecycleStage.view_type_to_stages(ViewType.ALL)),
                    name='runs_lifecycle_stage'),
]


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
            " by your SQL database.\nException content: %s", e)

    with op.batch_alter_table("runs", table_args=check_constraint_table_args) as batch_op:
        # Define a new "status" constraint via the SqlAlchemy `Enum` type. Specify
        # `native_enum=False` to create a check constraint rather than a
        # database-backend-dependent enum (see https://docs.sqlalchemy.org/en/13/core/
        # type_basics.html#sqlalchemy.types.En
        batch_op.alter_column("status",
                              existing_type=String,
                              type_=Enum(*new_run_statuses, create_constraint=True,
                                         constraint_name="status", native_enum=False))


def downgrade():
    # Omit downgrade logic for now - we don't currently provide users a command/API for
    # reverting a database migration, instead recommending that they take a database backup
    # before running the migration.
    pass

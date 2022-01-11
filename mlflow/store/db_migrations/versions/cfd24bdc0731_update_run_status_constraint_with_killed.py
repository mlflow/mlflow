"""Update run status constraint with killed

Revision ID: cfd24bdc0731
Revises: 89d4b8295536
Create Date: 2019-10-11 15:55:10.853449

"""
import alembic
from alembic import op
from mlflow.entities import RunStatus, ViewType
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.store.tracking.dbmodels.models import SqlRun, SourceTypes
from sqlalchemy import CheckConstraint, Enum
from packaging.version import Version

# revision identifiers, used by Alembic.
revision = "cfd24bdc0731"
down_revision = "2b4d017a5e9b"
branch_labels = None
depends_on = None

old_run_statuses = [
    RunStatus.to_string(RunStatus.SCHEDULED),
    RunStatus.to_string(RunStatus.FAILED),
    RunStatus.to_string(RunStatus.FINISHED),
    RunStatus.to_string(RunStatus.RUNNING),
]

new_run_statuses = [*old_run_statuses, RunStatus.to_string(RunStatus.KILLED)]

# Certain SQL backends (e.g., SQLite) do not preserve CHECK constraints during migrations.
# For these backends, CHECK constraints must be specified as table arguments. Here, we define
# the collection of CHECK constraints that should be preserved when performing the migration.
# The "status" constraint is excluded from this set because it is explicitly modified
# within the migration's `upgrade()` routine.
check_constraint_table_args = [
    CheckConstraint(SqlRun.source_type.in_(SourceTypes), name="source_type"),
    CheckConstraint(
        SqlRun.lifecycle_stage.in_(LifecycleStage.view_type_to_stages(ViewType.ALL)),
        name="runs_lifecycle_stage",
    ),
]


def upgrade():
    new_type = Enum(*new_run_statuses, create_constraint=True, native_enum=False)
    if Version(alembic.__version__) < Version("1.7.0"):
        with op.batch_alter_table("runs", table_args=check_constraint_table_args) as batch_op:
            # Transform the "status" column to an `Enum` and define a new check constraint. Specify
            # `native_enum=False` to create a check constraint rather than a
            # database-backend-dependent enum (see https://docs.sqlalchemy.org/en/13/core/
            # type_basics.html#sqlalchemy.types.Enum.params.native_enum)
            batch_op.alter_column("status", type_=new_type)
    else:
        # In alembic >= 1.7.0, `table_args` can be removed since CHECK constraints are preserved.
        with op.batch_alter_table("runs") as batch_op:
            existing_type = Enum(
                *old_run_statuses, create_constraint=True, native_enum=False, name="status"
            )
            batch_op.alter_column(
                "status",
                type_=new_type,
                # In alembic >= 1.7.0, `existing_type` is required to drop the existing CHECK
                # constraint on the status column.
                existing_type=existing_type,
            )


def downgrade():
    # Omit downgrade logic for now - we don't currently provide users a command/API for
    # reverting a database migration, instead recommending that they take a database backup
    # before running the migration.
    pass

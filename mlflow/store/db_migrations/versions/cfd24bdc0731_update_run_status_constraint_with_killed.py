"""Update run status constraint with killed

Revision ID: cfd24bdc0731
Revises: 89d4b8295536
Create Date: 2019-10-11 15:55:10.853449

"""
from alembic import op, context
from mlflow.entities import RunStatus, ViewType
from mlflow.entities.lifecycle_stage import LifecycleStage
from sqlalchemy import CheckConstraint, Enum
from sqlalchemy.engine.reflection import Inspector

from mlflow.store.tracking.dbmodels.models import SqlRun, SourceTypes


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


def _has_check_constraints():
    inspector = Inspector(context.get_bind().engine)
    constraints = [constraint['name']
                   for constraint in inspector.get_check_constraints("runs")]
    return "status" in constraints


def get_check_constraints():
    """
    Get all check constraint except status
    """
    inspector = Inspector(context.get_bind().engine)
    constraints = [CheckConstraint(constraint['sqltext'], name=constraint['name'])
                   for constraint in inspector.get_check_constraints("runs")
                   if constraint['name'] != 'status']
    return constraints

def expected_check_constraints():
    return (
        CheckConstraint(SqlRun.source_type.in_(SourceTypes), name='source_type'),
        CheckConstraint(SqlRun.status.in_(previous_run_statuses), name='status'),
        CheckConstraint(SqlRun.lifecycle_stage.in_(LifecycleStage.view_type_to_stages(ViewType.ALL)),
                        name='runs_lifecycle_stage'),
    )

def upgrade():
    print(get_check_constraints())

    # if _has_check_constraints():
    # with op.batch_alter_table("runs", table_args=get_check_constraints()) as batch_op:
    # with op.batch_alter_table("runs", copy_from=SqlRun.__table__, table_args={'extend_existing': True}) as batch_op:
    # with op.batch_alter_table("runs", copy_from=SqlRun.__table__, extend_existing=True) as batch_op:
    with op.batch_alter_table("runs", table_args=expected_check_constraints()) as batch_op:
    # with op.batch_alter_table("runs", table_args={'extend_existing': True}) as batch_op:
    # with op.batch_alter_table("runs") as batch_op:
        # Update the "status" constraint via the SqlAlchemy `Enum` type. Specify
        # `native_enum=False` to create a check constraint rather than a
        # database-backend-dependent enum (see https://docs.sqlalchemy.org/en/13/core/
        # type_basics.html#sqlalchemy.types.Enum.params.native_enum)
        batch_op.alter_column("status",
                              type_=Enum(*new_run_statuses, create_constraint=True,
                                         constraint_name="status", native_enum=False))


def downgrade():
    # Omit downgrade logic for now - we don't currently provide users a command/API for
    # reverting a database migration, instead recommending that they take a database backup
    # before running the migration.
    pass

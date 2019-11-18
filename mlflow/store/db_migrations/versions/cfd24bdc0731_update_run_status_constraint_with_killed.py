"""Update run status constraint with killed

Revision ID: cfd24bdc0731
Revises: 89d4b8295536
Create Date: 2019-10-11 15:55:10.853449

"""
from alembic import op, context
from mlflow.store.tracking.dbmodels.models import SqlRun
from mlflow.entities import RunStatus

# revision identifiers, used by Alembic.
revision = 'cfd24bdc0731'
down_revision = '2b4d017a5e9b'
branch_labels = None
depends_on = None


new_run_status = [
    RunStatus.to_string(RunStatus.SCHEDULED),
    RunStatus.to_string(RunStatus.FAILED),
    RunStatus.to_string(RunStatus.FINISHED),
    RunStatus.to_string(RunStatus.RUNNING),
    RunStatus.to_string(RunStatus.KILLED)
]

previous_run_status = [
    RunStatus.to_string(RunStatus.SCHEDULED),
    RunStatus.to_string(RunStatus.FAILED),
    RunStatus.to_string(RunStatus.FINISHED),
    RunStatus.to_string(RunStatus.RUNNING)
]


def _is_backend_sqlite():
    return context.get_bind().engine.url.get_backend_name() == 'sqlite'


def upgrade():
    with op.batch_alter_table("runs") as batch_op:
        if not _is_backend_sqlite():
            batch_op.drop_constraint(constraint_name='status', type_='check')
            batch_op.create_check_constraint(constraint_name='status',
                                             condition=SqlRun.status.in_(new_run_status))


def downgrade():
    with op.batch_alter_table("runs") as batch_op:
        if not _is_backend_sqlite():
            batch_op.drop_constraint(constraint_name='status', type_='check')
            batch_op.create_check_constraint(constraint_name='status',
                                             condition=SqlRun.status.in_(previous_run_status))

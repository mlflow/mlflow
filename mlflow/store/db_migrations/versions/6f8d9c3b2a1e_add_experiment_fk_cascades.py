"""Add ON DELETE CASCADE to `trace_info.experiment_id`.

Fixes https://github.com/mlflow/mlflow/issues/18781.

Create Date: 2026-07-09 21:16:54.445000

"""

from alembic import op

from mlflow.store.tracking.dbmodels.models import SqlExperiment, SqlTraceInfo

# revision identifiers, used by Alembic.
revision = "6f8d9c3b2a1e"
down_revision = "a8b9c0d1e2f3"
branch_labels = None
depends_on = None


_TRACE_INFO_EXPERIMENT_FK = "fk_trace_info_experiment_id"


def _alter_experiment_fk(ondelete=None):
    fk_kwargs = {"ondelete": ondelete} if ondelete else {}
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table(SqlTraceInfo.__tablename__, schema=None) as batch_op:
            batch_op.drop_constraint(_TRACE_INFO_EXPERIMENT_FK, type_="foreignkey")
            batch_op.create_foreign_key(
                _TRACE_INFO_EXPERIMENT_FK,
                SqlExperiment.__tablename__,
                ["experiment_id"],
                ["experiment_id"],
                **fk_kwargs,
            )
    else:
        op.drop_constraint(
            _TRACE_INFO_EXPERIMENT_FK,
            SqlTraceInfo.__tablename__,
            type_="foreignkey",
        )
        op.create_foreign_key(
            _TRACE_INFO_EXPERIMENT_FK,
            SqlTraceInfo.__tablename__,
            SqlExperiment.__tablename__,
            ["experiment_id"],
            ["experiment_id"],
            **fk_kwargs,
        )


def upgrade():
    _alter_experiment_fk(ondelete="CASCADE")


def downgrade():
    _alter_experiment_fk()

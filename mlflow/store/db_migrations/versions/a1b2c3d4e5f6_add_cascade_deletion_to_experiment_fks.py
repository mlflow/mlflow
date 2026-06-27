"""add cascade deletion to experiment_id foreign keys

Missing ON DELETE CASCADE on four experiment_id FK constraints:
  - trace_info.experiment_id
  - logged_model_metrics.experiment_id
  - logged_model_params.experiment_id
  - logged_model_tags.experiment_id

Create Date: 2025-05-14 00:00:00.000000

"""

from alembic import op

from mlflow.store.tracking.dbmodels.models import (
    SqlLoggedModelMetric,
    SqlLoggedModelParam,
    SqlLoggedModelTag,
    SqlTraceInfo,
)

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = "f5a4f2784254"
branch_labels = None
depends_on = None

# (table_name, fk_constraint_name, fk_column, ref_table, ref_column)
_FK_SPECS = [
    (
        SqlTraceInfo.__tablename__,
        "fk_trace_info_experiment_id",
        "experiment_id",
        "experiments",
        "experiment_id",
    ),
    (
        SqlLoggedModelMetric.__tablename__,
        "fk_logged_model_metrics_experiment_id",
        "experiment_id",
        "experiments",
        "experiment_id",
    ),
    (
        SqlLoggedModelParam.__tablename__,
        "fk_logged_model_params_experiment_id",
        "experiment_id",
        "experiments",
        "experiment_id",
    ),
    (
        SqlLoggedModelTag.__tablename__,
        "fk_logged_model_tags_experiment_id",
        "experiment_id",
        "experiments",
        "experiment_id",
    ),
]


def upgrade():
    for table, constraint_name, local_col, ref_table, ref_col in _FK_SPECS:
        # batch_alter_table is required for SQLite compatibility (SQLite does not
        # support ALTER TABLE ... DROP/ADD CONSTRAINT outside a batch operation).
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.drop_constraint(constraint_name, type_="foreignkey")
            batch_op.create_foreign_key(
                constraint_name,
                ref_table,
                [local_col],
                [ref_col],
                ondelete="CASCADE",
            )


def downgrade():
    # Intentionally left as a no-op; removing CASCADE is a safe no-op
    # (it only affects what happens when the referenced row is deleted).
    pass

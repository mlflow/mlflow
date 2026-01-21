"""add cascade deletion to logged_model tables experiment_id foreign keys

Create Date: 2026-01-21

"""

from alembic import op

from mlflow.store.tracking.dbmodels.models import (
    SqlLoggedModelMetric,
    SqlLoggedModelParam,
    SqlLoggedModelTag,
)

# revision identifiers, used by Alembic.
revision = "92d5b24e0351"
down_revision = "2c33131f4dae"
branch_labels = None
depends_on = None


def upgrade():
    tables_and_fks = [
        (SqlLoggedModelParam.__tablename__, "fk_logged_model_params_experiment_id"),
        (SqlLoggedModelTag.__tablename__, "fk_logged_model_tags_experiment_id"),
        (SqlLoggedModelMetric.__tablename__, "fk_logged_model_metrics_experiment_id"),
    ]
    for table, fk_constraint_name in tables_and_fks:
        # We have to use batch_alter_table as SQLite does not support
        # ALTER outside of a batch operation.
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.drop_constraint(fk_constraint_name, type_="foreignkey")
            batch_op.create_foreign_key(
                fk_constraint_name,
                "experiments",
                ["experiment_id"],
                ["experiment_id"],
                ondelete="CASCADE",
            )


def downgrade():
    pass

"""add cascading deletion to datasets from experiments

Revision ID: 0584bdc529eb
Revises: f5a4f2784254
Create Date: 2024-11-11 15:27:53.189685

"""
from alembic import op
import sqlalchemy as sa

from mlflow.store.tracking.dbmodels.models import SqlDataset


# revision identifiers, used by Alembic.
revision = '0584bdc529eb'
down_revision = 'f5a4f2784254'
branch_labels = None
depends_on = None


def upgrade():
    dialect_name = op.get_context().dialect.name
    if dialect_name == "sqlite":
        # Only way to drop unnamed fk constraint in sqllite
        # See https://alembic.sqlalchemy.org/en/latest/batch.html#dropping-unnamed-or-named-foreign-key-constraints
        with op.batch_alter_table(
            "datasets",
            schema=None,
            naming_convention={
                "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            },
        ) as batch_op:
            batch_op.drop_constraint("fk_datasets_experiment_id_experiments", type_="foreignkey")
            # Need to explicitly name the fk constraint with batch alter table
            batch_op.create_foreign_key(
                "fk_datasets_experiment_id_experiments",
                "experiments",
                ["experiment_id"],
                ["experiment_id"],
                ondelete="CASCADE",
            )
    else:
        if dialect_name == "postgresql":
            fk_constraint_name = "datasets_experiment_id_fkey"
        elif dialect_name == "mysql":
            fk_constraint_name = "datasets_ibfk_1"
        elif dialect_name == "mssql":
            # mssql fk constraint name at f5a4f2784254
            fk_constraint_name = "FK__datasets__experi__6477ECF3"

        # don't use batch alter table here so `create_foreign_key()` can be
        # called with `None` as the `constraint_name` and have it set
        # automatically based on the conventions of the sql flavor
        op.drop_constraint(fk_constraint_name, "datasets", type_="foreignkey")
        op.create_foreign_key(
            None,
            "datasets",
            "experiments",
            ["experiment_id"],
            ["experiment_id"],
            ondelete="CASCADE",
        )


def downgrade():
    pass

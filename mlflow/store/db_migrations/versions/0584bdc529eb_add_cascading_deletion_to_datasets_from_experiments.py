"""add cascading deletion to datasets from experiments

Revision ID: 0584bdc529eb
Revises: f5a4f2784254
Create Date: 2024-11-11 15:27:53.189685

"""
from alembic import op
import sqlalchemy as sa

from mlflow.store.tracking.dbmodels.models import SqlDataset, SqlExperiment


# revision identifiers, used by Alembic.
revision = '0584bdc529eb'
down_revision = 'f5a4f2784254'
branch_labels = None
depends_on = None


def upgrade():
    dialect_name = op.get_context().dialect.name

    # standardize the constraint to sqlite naming convention
    new_fk_constraint_name = f"fk_{SqlDataset.__tablename__}_experiment_id_{SqlExperiment.__tablename__}"

    if dialect_name == "sqlite":
        # Only way to drop unnamed fk constraint in sqllite
        # See https://alembic.sqlalchemy.org/en/latest/batch.html#dropping-unnamed-or-named-foreign-key-constraints
        with op.batch_alter_table(
            SqlDataset.__tablename__,
            schema=None,
            naming_convention={
                "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            },
        ) as batch_op:
            batch_op.drop_constraint(new_fk_constraint_name, type_="foreignkey")
            # Need to explicitly name the fk constraint with batch alter table
            batch_op.create_foreign_key(
                new_fk_constraint_name,
                SqlExperiment.__tablename__,
                ["experiment_id"],
                ["experiment_id"],
                ondelete="CASCADE",
            )
    else:
        if dialect_name == "postgresql":
            fk_constraint_name = f"{SqlDataset.__tablename__}_experiment_id_fkey"
        elif dialect_name == "mysql":
            fk_constraint_name = f"{SqlDataset.__tablename__}_ibfk_1"
        elif dialect_name == "mssql":
            # mssql fk constraint name at f5a4f2784254
            fk_constraint_name = f"FK__{SqlDataset.__tablename__}__experi__6477ECF3"

        op.drop_constraint(fk_constraint_name, SqlDataset.__tablename__, type_="foreignkey")
        op.create_foreign_key(
            new_fk_constraint_name,
            SqlDataset.__tablename__,
            SqlExperiment.__tablename__,
            ["experiment_id"],
            ["experiment_id"],
            ondelete="CASCADE",
        )


def downgrade():
    pass

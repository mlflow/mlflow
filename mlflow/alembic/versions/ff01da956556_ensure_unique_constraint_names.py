"""ensure_unique_constraint_names

Revision ID: ff01da956556
Revises: 
Create Date: 2019-05-18 22:58:06.487489

"""
from alembic import op
from sqlalchemy import column, CheckConstraint
from mlflow.store.dbmodels.initial_models import SqlExperiment, SqlRun

# revision identifiers, used by Alembic.
revision = 'ff01da956556'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Use batch mode so that we can run "ALTER TABLE" statements against SQLite
    # databases (see more info at https://alembic.sqlalchemy.org/en/latest/
    # batch.html#running-batch-migrations-for-sqlite-and-other-databases).
    # Also, we directly pass the schema of the table we're modifying to circumvent shortcomings
    # in Alembic's ability to reflect CHECK constraints, as described in
    # https://alembic.sqlalchemy.org/en/latest/batch.html#working-in-offline-mode
    with op.batch_alter_table("experiments", copy_from=SqlExperiment.__table__) as batch_op:
        batch_op.drop_constraint(constraint_name='lifecycle_stage', type_="check")
        batch_op.create_check_constraint(
            constraint_name="experiments_lifecycle_stage",
            condition=column('lifecycle_stage').in_(["active", "deleted"])
        )
    with op.batch_alter_table("runs", copy_from=SqlRun.__table__) as batch_op:
        batch_op.drop_constraint(constraint_name='lifecycle_stage', type_="check")
        batch_op.create_check_constraint(
            constraint_name="runs_lifecycle_stage",
            condition=column('lifecycle_stage').in_(["active", "deleted"])
        )


def downgrade():
    # Omit downgrade logic for now - we don't currently provide users a command/API for
    # reverting a database migration, instead recommending that they take a database backup
    # before running the migration.
    pass

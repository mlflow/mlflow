"""add cascade deletion to trace tables foreign keys

Revision ID: 5b0e9adcef9c
Revises: 867495a8f9d4
Create Date: 2024-05-22 17:44:24.597019

"""
from alembic import op
from mlflow.store.tracking.dbmodels.models import SqlTraceInfo, SqlTraceRequestMetadata, SqlTraceTag


# revision identifiers, used by Alembic.
revision = '5b0e9adcef9c'
down_revision = '867495a8f9d4'
branch_labels = None
depends_on = None


def upgrade():
    tables = [SqlTraceTag.__tablename__, SqlTraceRequestMetadata.__tablename__]
    for table in tables:
        fk_tag_constraint_name = f"fk_{table}_request_id"
        # We have to use batch_alter_table as SQLite does not support ALTER outside of a batch operation.
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.drop_constraint(fk_tag_constraint_name, type_="foreignkey")
            batch_op.create_foreign_key(
                fk_tag_constraint_name,
                SqlTraceInfo.__tablename__,
                ["request_id"],
                ["request_id"],
                # Add cascade deletion to the foreign key constraint. This is the only change in this migration.
                ondelete="CASCADE",
            )


def downgrade():
    pass

"""add unique constraint to input_uuid

Create Date: 2026-02-12 00:00:00.000000

"""

from alembic import op

from mlflow.store.tracking.dbmodels.models import SqlInput

# revision identifiers, used by Alembic.
revision = "9eae7ec868c3"
down_revision = "1b5f0d9ad7c1"
branch_labels = None
depends_on = None


def upgrade():
    """
    Add a unique constraint to the input_uuid column in the inputs table.
    This ensures input_uuid values are unique at the database level, which is
    required for the ORM-level foreign key relationship from input_tags to inputs
    and aligns with the application's expectation that input_uuid is a stable,
    unique identifier.
    """
    # Use batch mode for SQLite compatibility
    with op.batch_alter_table(SqlInput.__tablename__, schema=None) as batch_op:
        batch_op.create_unique_constraint(
            f"uq_{SqlInput.__tablename__}_input_uuid",
            ["input_uuid"],
        )


def downgrade():
    """
    Remove the unique constraint from the input_uuid column.
    Note: This downgrade may fail if there are input_tags referencing inputs
    via the ORM-level foreign key, as the ORM expects the referenced column
    to be unique.
    """
    # Use batch mode for SQLite compatibility
    with op.batch_alter_table(SqlInput.__tablename__, schema=None) as batch_op:
        batch_op.drop_constraint(
            f"uq_{SqlInput.__tablename__}_input_uuid",
            type_="unique",
        )

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
    This is needed to properly support the foreign key relationship from
    input_tags to inputs.
    """
    # Use batch mode for SQLite compatibility
    with op.batch_alter_table(SqlInput.__tablename__, schema=None) as batch_op:
        batch_op.create_unique_constraint(
            f"uq_{SqlInput.__tablename__}_input_uuid",
            ["input_uuid"],
        )


def downgrade():
    pass

"""increase run tag value limit to 8000

Create Date: 2024-09-18 08:53:51.552934

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "f5a4f2784254"
down_revision = "4465047574b1"
branch_labels = None
depends_on = None


def upgrade():
    # Use batch mode so that we can run "ALTER TABLE" statements against SQLite
    # databases (see more info at https://alembic.sqlalchemy.org/en/latest/
    # batch.html#running-batch-migrations-for-sqlite-and-other-databases)
    # We specify existing_type, existing_nullable, existing_server_default
    # because MySQL alter column statements require a full column description.
    with op.batch_alter_table("tags") as batch_op:
        batch_op.alter_column(
            "value",
            existing_type=sa.String(5000),
            type_=sa.String(8000),
            existing_nullable=True,
            existing_server_default=None,
        )


def downgrade():
    pass

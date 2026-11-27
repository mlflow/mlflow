"""increase experiment tag value limit to 20000

Create Date: 2026-08-13 16:36:37.291014

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mssql import NVARCHAR
from sqlalchemy.dialects.mysql import MEDIUMTEXT

# revision identifiers, used by Alembic.
revision = "17e22815139b"
down_revision = "1971f2d9a75b"
branch_labels = None
depends_on = None


def upgrade():
    # Use batch mode so that we can run "ALTER TABLE" statements against SQLite
    # databases (see more info at https://alembic.sqlalchemy.org/en/latest/
    # batch.html#running-batch-migrations-for-sqlite-and-other-databases)
    # We specify existing_type, existing_nullable, existing_server_default
    # because MySQL alter column statements require a full column description.
    # MEDIUMTEXT provides enough capacity for 20,000 characters with the MySQL utf8mb4 charset.
    # NVARCHAR(max) preserves multibyte characters on SQL Server.
    with op.batch_alter_table("experiment_tags") as batch_op:
        batch_op.alter_column(
            "value",
            existing_type=sa.String(5000),
            type_=(
                sa.Text().with_variant(MEDIUMTEXT, "mysql").with_variant(NVARCHAR(None), "mssql")
            ),
            existing_nullable=True,
            existing_server_default=None,
        )


def downgrade():
    pass

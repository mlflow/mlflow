"""allow nulls for run_id

Create Date: 2020-12-02 12:14:35.220815

"""

import sqlalchemy as sa
from alembic import op

from mlflow.store.model_registry.dbmodels.models import SqlModelVersion

# revision identifiers, used by Alembic.
revision = "a8c4a736bde6"
down_revision = "84291f40a231"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table(SqlModelVersion.__tablename__) as batch_op:
        batch_op.alter_column("run_id", nullable=True, existing_type=sa.VARCHAR(32))


def downgrade():
    pass

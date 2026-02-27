"""add registered model tags table

Create Date: 2020-06-26 13:30:00.290154

"""

import sqlalchemy as sa
from alembic import op

from mlflow.store.model_registry.dbmodels.models import SqlRegisteredModelTag

# revision identifiers, used by Alembic.
revision = "728d730b5ebd"
down_revision = "0a8213491aaa"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        SqlRegisteredModelTag.__tablename__,
        sa.Column("key", sa.String(length=250), primary_key=True, nullable=False),
        sa.Column("value", sa.String(length=5000)),
        sa.Column(
            "name",
            sa.String(length=256),
            sa.ForeignKey("registered_models.name", onupdate="cascade"),
            primary_key=True,
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("key", "name", name="registered_model_tag_pk"),
    )


def downgrade():
    pass

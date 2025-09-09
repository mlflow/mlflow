"""add model version tags table

Create Date: 2020-06-26 13:30:27.611086

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
from mlflow.store.model_registry.dbmodels.models import SqlModelVersionTag

revision = "27a6a02d2cf1"
down_revision = "728d730b5ebd"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        SqlModelVersionTag.__tablename__,
        sa.Column("key", sa.String(length=250), primary_key=True, nullable=False),
        sa.Column("value", sa.String(length=5000)),
        sa.Column("name", sa.String(length=256), primary_key=True, nullable=False),
        sa.Column("version", sa.Integer(), primary_key=True, nullable=False),
        sa.ForeignKeyConstraint(
            ("name", "version"),
            ("model_versions.name", "model_versions.version"),
            onupdate="cascade",
        ),
        sa.PrimaryKeyConstraint("key", "name", "version", name="model_version_tag_pk"),
    )


def downgrade():
    pass

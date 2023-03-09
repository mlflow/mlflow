"""Add Model Aliases table

Revision ID: 29599da4828d
Revises: 97727af70f4d
Create Date: 2023-03-08 13:22:37.174258

"""
from alembic import op
import sqlalchemy as sa
from mlflow.store.model_registry.dbmodels.models import SqlRegisteredModelAlias

# revision identifiers, used by Alembic.

revision = "29599da4828d"
down_revision = "97727af70f4d"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        SqlRegisteredModelAlias.__tablename__,
        sa.Column("name", sa.String(length=256), primary_key=True, nullable=False),
        sa.Column("alias", sa.String(length=250), primary_key=True, nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("name", "alias", name="registered_model_alias_pk"),
        sa.ForeignKeyConstraint(
            ("name", "version"),
            ("model_versions.name", "model_versions.version"),
            onupdate="cascade",
            name="registered_model_alias_name_version_fkey",
        ),
    )


def downgrade():
    pass

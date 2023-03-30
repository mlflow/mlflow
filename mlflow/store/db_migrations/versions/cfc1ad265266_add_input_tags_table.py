"""add input_tags table

Revision ID: cfc1ad265266
Revises: 30da9e9d8245
Create Date: 2023-03-30 01:26:59.311031

"""
from alembic import op
import sqlalchemy as sa
from mlflow.store.tracking.dbmodels.models import SqlInputTag


# revision identifiers, used by Alembic.
revision = "cfc1ad265266"
down_revision = "30da9e9d8245"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        SqlInputTag.__tablename__,
        sa.Column(
            "input_uuid",
            sa.String(length=36),
            sa.ForeignKey("inputs.input_uuid"),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("name", sa.String(length=255), primary_key=True, nullable=False),
        sa.Column("value", sa.String(length=500), nullable=False),
        sa.PrimaryKeyConstraint("input_uuid", "name", name="input_tags_pk"),
    )


def downgrade():
    pass

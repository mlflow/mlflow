"""Add webhooks and webhook_events tables

Revision ID: 1a0cddfcaa16
Revises: cbc13b556ace
Create Date: 2025-07-07 23:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

from mlflow.store.model_registry.dbmodels.models import SqlWebhook, SqlWebhookEvent

# revision identifiers, used by Alembic.
revision = "1a0cddfcaa16"
down_revision = "cbc13b556ace"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        SqlWebhook.__tablename__,
        sa.Column("webhook_id", sa.String(length=256), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("description", sa.String(length=1000), nullable=True),
        sa.Column("url", sa.String(length=500), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="ACTIVE"),
        sa.Column("secret", sa.String(length=1000), nullable=True),  # Stored as encrypted text
        sa.Column("creation_timestamp", sa.BigInteger(), nullable=True),
        sa.Column("last_updated_timestamp", sa.BigInteger(), nullable=True),
        sa.Column("deleted_timestamp", sa.BigInteger(), nullable=True),  # For soft deletes
        sa.PrimaryKeyConstraint("webhook_id", name="webhook_pk"),
    )

    op.create_table(
        SqlWebhookEvent.__tablename__,
        sa.Column("webhook_id", sa.String(length=256), nullable=False),
        sa.Column("event", sa.String(length=50), nullable=False),
        sa.ForeignKeyConstraint(
            ["webhook_id"], [f"{SqlWebhook.__tablename__}.webhook_id"], ondelete="cascade"
        ),
        sa.PrimaryKeyConstraint("webhook_id", "event", name="webhook_event_pk"),
    )


def downgrade():
    # Drop webhook_events table first due to foreign key constraint
    op.drop_table(SqlWebhookEvent.__tablename__)
    op.drop_table(SqlWebhook.__tablename__)

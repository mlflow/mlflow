"""add webhooks table to db

Revision ID: e396268937fc
Revises: 0584bdc529eb
Create Date: 2025-03-26 01:50:20.419534

"""

import logging
import time

from alembic import op
from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    PrimaryKeyConstraint,
    String,
    orm,
)

from mlflow.store.model_registry.dbmodels.models import SqlWebhook

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

# revision identifiers, used by Alembic.
revision = "e396268937fc"
down_revision = "0584bdc529eb"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    _logger.info("Adding webhooks table to database.")

    op.create_table(
        SqlWebhook.__tablename__,
        Column("name", String(256), unique=True, nullable=False),
        Column("creation_time", BigInteger, default=lambda: int(time.time() * 1000)),
        Column("last_updated_time", BigInteger, nullable=True, default=None),
        Column("description", String(5000), nullable=True),
        Column("url", String(2048), nullable=False),
        Column("event_trigger", String(10), nullable=False),
        Column("key", String(256), nullable=False),
        Column("value", String(5000), nullable=True),
        Column("headers", JSON, nullable=True),
        Column("payload", JSON, nullable=True),
        PrimaryKeyConstraint("name", name="webhook_pk"),
    )

    session.commit()

    _logger.info("Migration complete!")


def downgrade():
    op.drop_table(SqlWebhook.__tablename__)

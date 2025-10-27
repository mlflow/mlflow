"""add model registry tables to db

Create Date: 2019-10-14 12:20:12.874424

"""

import logging
import time

from alembic import op
from sqlalchemy import (
    BigInteger,
    Column,
    ForeignKey,
    Integer,
    PrimaryKeyConstraint,
    String,
    orm,
)

from mlflow.entities.model_registry.model_version_stages import STAGE_NONE
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.store.model_registry.dbmodels.models import SqlModelVersion, SqlRegisteredModel

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

# revision identifiers, used by Alembic.
revision = "2b4d017a5e9b"
down_revision = "89d4b8295536"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    _logger.info("Adding registered_models and model_versions tables to database.")

    op.create_table(
        SqlRegisteredModel.__tablename__,
        Column("name", String(256), unique=True, nullable=False),
        Column("creation_time", BigInteger, default=lambda: int(time.time() * 1000)),
        Column("last_updated_time", BigInteger, nullable=True, default=None),
        Column("description", String(5000), nullable=True),
        PrimaryKeyConstraint("name", name="registered_model_pk"),
    )

    op.create_table(
        SqlModelVersion.__tablename__,
        Column("name", String(256), ForeignKey("registered_models.name", onupdate="cascade")),
        Column("version", Integer, nullable=False),
        Column("creation_time", BigInteger, default=lambda: int(time.time() * 1000)),
        Column("last_updated_time", BigInteger, nullable=True, default=None),
        Column("description", String(5000), nullable=True),
        Column("user_id", String(256), nullable=True, default=None),
        Column("current_stage", String(20), default=STAGE_NONE),
        Column("source", String(500), nullable=True, default=None),
        Column("run_id", String(32), nullable=False),
        Column(
            "status", String(20), default=ModelVersionStatus.to_string(ModelVersionStatus.READY)
        ),
        Column("status_message", String(500), nullable=True, default=None),
        PrimaryKeyConstraint("name", "version", name="model_version_pk"),
    )

    session.commit()

    _logger.info("Migration complete!")


def downgrade():
    op.drop_table(SqlRegisteredModel.__tablename__)
    op.drop_table(SqlModelVersion.__tablename__)

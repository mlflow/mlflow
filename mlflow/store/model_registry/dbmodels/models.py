from cryptography.fernet import Fernet
from sqlalchemy import (
    BigInteger,
    Column,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    TypeDecorator,
)
from sqlalchemy.orm import backref, relationship

from mlflow.entities.model_registry import (
    ModelVersion,
    ModelVersionTag,
    RegisteredModel,
    RegisteredModelAlias,
    RegisteredModelTag,
)
from mlflow.entities.model_registry.model_version_stages import STAGE_DELETED_INTERNAL, STAGE_NONE
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.entities.webhook import (
    Webhook,
    WebhookAction,
    WebhookEntity,
    WebhookEvent,
    WebhookStatus,
)
from mlflow.environment_variables import MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY
from mlflow.store.db.base_sql_model import Base
from mlflow.utils.time import get_current_time_millis


class SqlRegisteredModel(Base):
    __tablename__ = "registered_models"

    name = Column(String(256), unique=True, nullable=False)

    creation_time = Column(BigInteger, default=get_current_time_millis)

    last_updated_time = Column(BigInteger, nullable=True, default=None)

    description = Column(String(5000), nullable=True)

    __table_args__ = (PrimaryKeyConstraint("name", name="registered_model_pk"),)

    def __repr__(self):
        return (
            f"<SqlRegisteredModel ({self.name}, {self.description}, "
            f"{self.creation_time}, {self.last_updated_time})>"
        )

    def to_mlflow_entity(self):
        # SqlRegisteredModel has backref to all "model_versions". Filter latest for each stage.
        latest_versions = {}
        for mv in self.model_versions:
            stage = mv.current_stage
            if stage != STAGE_DELETED_INTERNAL and (
                stage not in latest_versions or latest_versions[stage].version < mv.version
            ):
                latest_versions[stage] = mv
        return RegisteredModel(
            self.name,
            self.creation_time,
            self.last_updated_time,
            self.description,
            [mvd.to_mlflow_entity() for mvd in latest_versions.values()],
            [tag.to_mlflow_entity() for tag in self.registered_model_tags],
            [alias.to_mlflow_entity() for alias in self.registered_model_aliases],
        )


class SqlModelVersion(Base):
    __tablename__ = "model_versions"

    name = Column(String(256), ForeignKey("registered_models.name", onupdate="cascade"))

    version = Column(Integer, nullable=False)

    creation_time = Column(BigInteger, default=get_current_time_millis)

    last_updated_time = Column(BigInteger, nullable=True, default=None)

    description = Column(String(5000), nullable=True)

    user_id = Column(String(256), nullable=True, default=None)

    current_stage = Column(String(20), default=STAGE_NONE)

    source = Column(String(500), nullable=True, default=None)

    storage_location = Column(String(500), nullable=True, default=None)

    run_id = Column(String(32), nullable=True, default=None)

    run_link = Column(String(500), nullable=True, default=None)

    status = Column(String(20), default=ModelVersionStatus.to_string(ModelVersionStatus.READY))

    status_message = Column(String(500), nullable=True, default=None)

    # linked entities
    registered_model = relationship(
        "SqlRegisteredModel", backref=backref("model_versions", cascade="all")
    )

    __table_args__ = (PrimaryKeyConstraint("name", "version", name="model_version_pk"),)

    # entity mappers
    def to_mlflow_entity(self):
        return ModelVersion(
            self.name,
            self.version,
            self.creation_time,
            self.last_updated_time,
            self.description,
            self.user_id,
            self.current_stage,
            self.source,
            self.run_id,
            self.status,
            self.status_message,
            [tag.to_mlflow_entity() for tag in self.model_version_tags],
            self.run_link,
            [],
        )


class SqlRegisteredModelTag(Base):
    __tablename__ = "registered_model_tags"

    name = Column(String(256), ForeignKey("registered_models.name", onupdate="cascade"))

    key = Column(String(250), nullable=False)

    value = Column(String(5000), nullable=True)

    # linked entities
    registered_model = relationship(
        "SqlRegisteredModel", backref=backref("registered_model_tags", cascade="all")
    )

    __table_args__ = (PrimaryKeyConstraint("key", "name", name="registered_model_tag_pk"),)

    def __repr__(self):
        return f"<SqlRegisteredModelTag ({self.name}, {self.key}, {self.value})>"

    # entity mappers
    def to_mlflow_entity(self):
        return RegisteredModelTag(self.key, self.value)


class SqlModelVersionTag(Base):
    __tablename__ = "model_version_tags"

    name = Column(String(256))

    version = Column(Integer)

    key = Column(String(250), nullable=False)

    value = Column(Text, nullable=True)

    # linked entities
    model_version = relationship(
        "SqlModelVersion",
        foreign_keys=[name, version],
        backref=backref("model_version_tags", cascade="all"),
    )

    __table_args__ = (
        PrimaryKeyConstraint("key", "name", "version", name="model_version_tag_pk"),
        ForeignKeyConstraint(
            ("name", "version"),
            ("model_versions.name", "model_versions.version"),
            onupdate="cascade",
        ),
    )

    def __repr__(self):
        return f"<SqlModelVersionTag ({self.name}, {self.version}, {self.key}, {self.value})>"

    # entity mappers
    def to_mlflow_entity(self):
        return ModelVersionTag(self.key, self.value)


class SqlRegisteredModelAlias(Base):
    __tablename__ = "registered_model_aliases"
    name = Column(
        String(256),
        ForeignKey(
            "registered_models.name",
            onupdate="cascade",
            ondelete="cascade",
            name="registered_model_alias_name_fkey",
        ),
    )
    alias = Column(String(256), nullable=False)
    version = Column(Integer, nullable=False)

    # linked entities
    registered_model = relationship(
        "SqlRegisteredModel", backref=backref("registered_model_aliases", cascade="all")
    )

    __table_args__ = (PrimaryKeyConstraint("name", "alias", name="registered_model_alias_pk"),)

    def __repr__(self):
        return f"<SqlRegisteredModelAlias ({self.name}, {self.alias}, {self.version})>"

    # entity mappers
    def to_mlflow_entity(self):
        return RegisteredModelAlias(self.alias, self.version)


class EncryptedString(TypeDecorator):
    """
    A custom SQLAlchemy type that encrypts data before storing in the database
    and decrypts it when retrieving.
    """

    impl = String(1000)
    cache_ok = True

    def __init__(self):
        super().__init__()
        # Get encryption key from environment variable or generate one
        # In production, this should come from a secure key management service
        encryption_key = MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY.get() or Fernet.generate_key()
        self.cipher = Fernet(encryption_key)

    def process_bind_param(self, value, dialect):
        if value is not None:
            return self.cipher.encrypt(value.encode()).decode()
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return self.cipher.decrypt(value.encode()).decode()
        return value


class SqlWebhook(Base):
    __tablename__ = "webhooks"

    webhook_id = Column(String(256), nullable=False)
    name = Column(String(256), nullable=False)
    description = Column(String(1000), nullable=True)
    url = Column(String(500), nullable=False)
    status = Column(String(20), nullable=False, default="ACTIVE")
    secret = Column(EncryptedString(), nullable=True)  # Encrypted storage for HMAC secret
    creation_timestamp = Column(BigInteger, default=get_current_time_millis)
    last_updated_timestamp = Column(BigInteger, nullable=True, default=None)
    deleted_timestamp = Column(BigInteger, nullable=True, default=None)  # For soft deletes

    __table_args__ = (
        PrimaryKeyConstraint("webhook_id", name="webhook_pk"),
        Index("idx_webhooks_status", "status"),
        Index("idx_webhooks_name", "name"),
    )

    def __repr__(self):
        return (
            f"<SqlWebhook ({self.webhook_id}, {self.name}, {self.url}, "
            f"{self.status}, {self.creation_timestamp})>"
        )

    def to_mlflow_entity(self):
        return Webhook(
            webhook_id=self.webhook_id,
            name=self.name,
            url=self.url,
            events=[we.to_mlflow_entity() for we in self.webhook_events],
            creation_timestamp=self.creation_timestamp,
            last_updated_timestamp=self.last_updated_timestamp,
            description=self.description,
            status=WebhookStatus(self.status),
            secret=self.secret,
        )


class SqlWebhookEvent(Base):
    __tablename__ = "webhook_events"

    webhook_id = Column(String(256), ForeignKey("webhooks.webhook_id", ondelete="cascade"))
    entity = Column(String(50), nullable=False)
    action = Column(String(50), nullable=False)

    # Relationship
    webhook = relationship(
        "SqlWebhook", backref=backref("webhook_events", cascade="all, delete-orphan")
    )

    __table_args__ = (
        PrimaryKeyConstraint("webhook_id", "entity", "action", name="webhook_event_pk"),
        Index("idx_webhook_events_entity", "entity"),
        Index("idx_webhook_events_action", "action"),
        Index("idx_webhook_events_entity_action", "entity", "action"),
    )

    def __repr__(self):
        return f"<SqlWebhookEvent ({self.webhook_id}, {self.entity}, {self.action})>"

    def to_mlflow_entity(self):
        return WebhookEvent(entity=WebhookEntity(self.entity), action=WebhookAction(self.action))

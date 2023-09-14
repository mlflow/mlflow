from sqlalchemy import (
    BigInteger,
    Column,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    PrimaryKeyConstraint,
    String,
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

    value = Column(String(5000), nullable=True)

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

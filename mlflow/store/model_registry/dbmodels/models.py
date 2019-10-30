import time

from sqlalchemy import (
    Column, String, ForeignKey, Integer, BigInteger, PrimaryKeyConstraint)
from sqlalchemy.orm import relationship, backref

from mlflow.entities.model_registry import (RegisteredModel, RegisteredModelDetailed,
                                            ModelVersion, ModelVersionDetailed)
from mlflow.entities.model_registry.model_version_stages import STAGE_NONE, STAGE_DELETED_INTERNAL
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.store.db.base_sql_model import Base


class SqlRegisteredModel(Base):
    __tablename__ = 'registered_models'

    name = Column(String(256), unique=True, nullable=False)

    creation_time = Column(BigInteger, default=lambda: int(time.time() * 1000))

    last_updated_time = Column(BigInteger, nullable=True, default=None)

    description = Column(String(5000), nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint('name', name='registered_model_pk'),
    )

    def __repr__(self):
        return '<SqlRegisteredModel ({}, {}, {}, {})>'.format(self.name, self.description,
                                                              self.creation_time,
                                                              self.last_updated_time)

    # entity mappers
    def to_mlflow_entity(self):
        return RegisteredModel(self.name)

    def to_mlflow_detailed_entity(self):
        # SqlRegisteredModel has backref to all "model_versions". Filter latest for each stage.
        latest_versions = {}
        for mv in self.model_versions:
            stage = mv.current_stage
            if stage != STAGE_DELETED_INTERNAL and (stage not in latest_versions or
                                                    latest_versions[stage].version < mv.version):
                latest_versions[stage] = mv
        return RegisteredModelDetailed(self.name, self.creation_time, self.last_updated_time,
                                       self.description,
                                       [mvd.to_mlflow_detailed_entity()
                                        for mvd in latest_versions.values()])


class SqlModelVersion(Base):
    __tablename__ = 'model_versions'

    name = Column(String(256), ForeignKey('registered_models.name', onupdate='cascade'))

    version = Column(Integer, nullable=False)

    creation_time = Column(BigInteger, default=lambda: int(time.time() * 1000))

    last_updated_time = Column(BigInteger, nullable=True, default=None)

    description = Column(String(5000), nullable=True)

    user_id = Column(String(256), nullable=True, default=None)

    current_stage = Column(String(20), default=STAGE_NONE)

    source = Column(String(500), nullable=True, default=None)

    run_id = Column(String(32), nullable=False)

    status = Column(String(20),
                    default=ModelVersionStatus.to_string(ModelVersionStatus.READY))

    status_message = Column(String(500), nullable=True, default=None)

    # linked entities
    registered_model = relationship('SqlRegisteredModel',
                                    backref=backref('model_versions',
                                                    cascade='all'))

    __table_args__ = (
        PrimaryKeyConstraint('name', 'version', name='model_version_pk'),
    )

    # entity mappers
    def to_mlflow_entity(self):
        return ModelVersion(self.registered_model.to_mlflow_entity(), self.version)

    def to_mlflow_detailed_entity(self):
        return ModelVersionDetailed(self.registered_model.to_mlflow_entity(), self.version,
                                    self.creation_time, self.last_updated_time, self.description,
                                    self.user_id, self.current_stage, self.source, self.run_id,
                                    self.status, self.status_message)

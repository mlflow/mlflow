import enum
import time
import sqlalchemy
from sqlalchemy import Column, Integer, Text, String, Float, Enum, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from mlflow.entities import Experiment, ViewType, RunTag, Metric, Param, RunData

Base = declarative_base()

# TODO: Add mappings from the sql models to the mlflow entities as functions
# in  each model


class ViewTypeEnum(enum.Enum):
    ACTIVE_ONLY = ViewType.ACTIVE_ONLY
    DELETED_ONLY = ViewType.DELETED_ONLY
    ALL = ViewType.ALL


class EntityMixin(object):
    def to_mlflow_entity(self):
        if not hasattr(self, '__entity__'):
            raise Exception(
                'sqlalchemy model <{}> needs __entity__ set'.format(self.__class__.__name__))

        if not hasattr(self, '__properties__'):
            raise Exception(
                'sqlalchemy model <{}> needs __properties__ set'.format(self.__class__.__name__))

        # create dict of kwargs properties for entity and return the intialized entity
        config = {k: getattr(self, k) for k in self.__properties__}

        return self.__entity__.from_dictionary(config)


class SqlExperiment(Base, EntityMixin):
    __tablename__ = 'experiments'
    __entity__ = Experiment
    __properties__ = Experiment._properties()
    experiment_id = Column(Integer, primary_key=True)
    name = Column(String(256), unique=True, nullable=False)
    artifact_location = Column(Text, nullable=True)
    lifecycle_stage = Column(Enum(ViewTypeEnum),
                             default=ViewTypeEnum.ACTIVE_ONLY)

    def __repr__(self):
        return '<SqlExperiment ({}, {})>'.format(self.experiment_id, self.name)


class SqlRunTag(Base, EntityMixin):
    __tablename__ = 'run_tag'
    __entity__ = RunTag
    __properties__ = RunTag._properties()
    id = Column(Integer, primary_key=True)
    key = Column(Text, nullable=False)
    value = Column(Text, nullable=True)

    def __repr__(self):
        return '<SqlRunTag({}, {})>'.format(self.key, self.value)


class SqlMetric(Base, EntityMixin):
    __tablename__ = 'metric'
    __entity__ = Metric
    __properties__ = Metric._properties()
    id = Column(Integer, primary_key=True)
    key = Column(Text, nullable=False)
    value = Column(Float, nullable=False)
    timestamp = Column(Integer, default=int(time.time()))
    run_data_id = Column(Integer, ForeignKey('run_data.id'))
    run_data = sqlalchemy.orm.relationship('SqlRunData', backref='metrics')

    def __repr__(self):
        return '<SqlMetric({}, {})>'.format(self.key, self.value)


class SqlParam(Base, EntityMixin):
    __tablename__ = 'param'
    __entity__ = Param
    __properties__ = Param._properties()
    id = Column(Integer, primary_key=True)
    key = Column(Text, nullable=False)
    value = Column(Text, nullable=False)
    run_data_id = Column(Integer, ForeignKey('run_data.id'))
    run_data = sqlalchemy.orm.relationship('SqlRunData', backref='params')

    def __repr__(self):
        return '<SqlParam({}, {})>'.format(self.key, self.value)


class SqlRunData(Base, EntityMixin):
    __tablename__ = 'run_data'
    __entity__ = RunData
    __properties__ = RunData._properties()
    id = Column(Integer, primary_key=True)

    def __repr__(self):
        return '<SqlRunData({})>'.format(self.id)

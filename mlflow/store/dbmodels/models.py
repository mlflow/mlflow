import enum
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from mlflow.entities import Experiment, ViewType, RunTag

Base = declarative_base()

# TODO: Add mappings from the sql models to the mlflow entities as functions
# in  each model


class ViewTypeEnum(enum.Enum):
    ACTIVE_ONLY = ViewType.ACTIVE_ONLY
    DELETED_ONLY = ViewType.DELETED_ONLY
    ALL = ViewType.ALL


class EntityMapping(object):
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


class SqlExperiment(Base, EntityMapping):
    __tablename__ = 'experiments'
    __entity__ = Experiment
    __properties__ = Experiment._properties()
    experiment_id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    name = sqlalchemy.Column(sqlalchemy.String(256), unique=True, nullable=False)
    artifact_location = sqlalchemy.Column(sqlalchemy.Text, nullable=True)
    lifecycle_stage = sqlalchemy.Column(sqlalchemy.Enum(ViewTypeEnum),
                                        default=ViewTypeEnum.ACTIVE_ONLY)

    def __repr__(self):
        return '<SqlExperiment ({}, {})>'.format(self.experiment_id, self.name)


class SqlRunTag(Base, EntityMapping):
    __tablename__ = 'run_tag'
    __entity__ = RunTag
    __properties__ = RunTag._properties()
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    key = sqlalchemy.Column(sqlalchemy.TEXT, nullable=False)
    value = sqlalchemy.Column(sqlalchemy.TEXT, nullable=True)

    def __repr__(self):
        return '<SqlRunTag({}, {})>'.format(self.key, self.value)



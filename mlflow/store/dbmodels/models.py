import time
import uuid
import os
from sqlalchemy.orm import relationship, backref
from sqlalchemy import Column, Text, String, Float, ForeignKey, Integer, CheckConstraint
from sqlalchemy.ext.declarative import declarative_base
from mlflow.entities import Experiment, RunTag, Metric, Param, RunData, RunInfo,\
    SourceType, RunStatus, Run

Base = declarative_base()

# TODO: UPdate the types for better performance
# TODO: Create custom column type for run_uuid


def _get_user_id():
    try:
        import pwd
        return pwd.getpwuid(os.getuid())[0]
    except ImportError:
        return 'Unknown'


ExperimentLifecycleStages = [
    Experiment.ACTIVE_LIFECYCLE,
    Experiment.DELETED_LIFECYCLE
]

SourceTypes = [
    SourceType.NOTEBOOK,
    SourceType.JOB,
    SourceType.LOCAL,
    SourceType.UNKNOWN,
    SourceType.PROJECT
]

RunStatusTypes = [
    RunStatus.SCHEDULED,
    RunStatus.FAILED,
    RunStatus.FINISHED,
    RunStatus.RUNNING
]

LifecycleStageTypes = [
    RunInfo.ACTIVE_LIFECYCLE,
    RunInfo.DELETED_LIFECYCLE
]


class EntityMixin(object):
    """
    Converts alchemy models to mlflow entities
    """

    def __init__(self):
        # Propeties is the required values
        # Entity is the mlflow class like Metric etc..
        self.__properties__ = None
        self.__entity__ = None

    def _validate(self):
        if not hasattr(self, '__entity__') and self.__entity__ is not None:
            raise Exception(
                'sqlalchemy model <{}> needs __entity__ set'.format(self.__class__.__name__))

        if not hasattr(self, '__properties__') and self.__entity__ is not None:
            raise Exception(
                'sqlalchemy model <{}> needs __properties__ set'.format(self.__class__.__name__))

    def to_mlflow_entity(self):
        self._validate()

        # create dict of kwargs properties for entity and return the intialized entity
        config = {}
        for k in self.__properties__:
            # check if its mlflow entity and build it
            obj = getattr(self, k)
            try:
                config[k] = obj.to_mlflow_entity()
            except AttributeError:
                if k in ['metrics', 'params', 'tags']:
                    # these are list so
                    obj = [v.to_mlflow_entity() for v in obj]

                config[k] = obj

        return self.__entity__(**config)


class SqlExperiment(Base, EntityMixin):
    __tablename__ = 'experiments'
    __entity__ = Experiment
    __properties__ = Experiment._properties()
    experiment_id = Column(Integer, primary_key=True)
    name = Column(String(256), unique=True, nullable=False)
    artifact_location = Column(Text, nullable=True)
    lifecycle_stage = Column(Integer, default=Experiment.ACTIVE_LIFECYCLE)

    __table_args__ = (
        CheckConstraint(
            lifecycle_stage.in_(ExperimentLifecycleStages), name='lifecycle_stage'),
    )

    def __repr__(self):
        return '<SqlExperiment ({}, {})>'.format(self.experiment_id, self.name)


class SqlRunTag(Base, EntityMixin):
    __tablename__ = 'run_tag'
    __entity__ = RunTag
    __properties__ = RunTag._properties()
    id = Column(Integer, primary_key=True)
    run_data_id = Column(Integer, ForeignKey('run_data.id'))
    run_data = relationship('SqlRunData', backref='tags')
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
    run_data = relationship('SqlRunData', backref=backref('metrics', cascade='all,delete'))

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
    run_data = relationship('SqlRunData', backref=backref('params', cascade='all,delete'))

    def __repr__(self):
        return '<SqlParam({}, {})>'.format(self.key, self.value)


class SqlRunData(Base, EntityMixin):
    __tablename__ = 'run_data'
    __entity__ = RunData
    __properties__ = RunData._properties()
    id = Column(Integer, primary_key=True)

    def __repr__(self):
        return '<SqlRunData({})>'.format(self.id)


class SqlRunInfo(Base, EntityMixin):
    __tablename__ = 'run_info'
    __entity__ = RunInfo
    __properties__ = RunInfo._properties()
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiments.experiment_id'))
    run_uuid = uuid.uuid4().hex
    name = Column(Text)
    # source_type = Column(Enum(SourceTypeEnum), default=SourceTypeEnum.LOCAL)
    source_type = Column(Integer, default=SourceType.LOCAL)
    source_name = Column(String(256))
    entry_point_name = Column(Text)
    user_id = Column(Text, default=_get_user_id(), nullable=False)
    status = Column(Integer, default=RunStatus.SCHEDULED)
    start_time = Column(Integer, default=int(time.time()))
    end_time = Column(Integer, nullable=True, default=None)
    source_version = Column(Text)
    lifecycle_stage = Column(Integer, default=RunInfo.ACTIVE_LIFECYCLE)
    artifact_uri = Column(Text, default=None)

    __table_args__ = (
        CheckConstraint(source_type.in_(SourceTypes), name='source_type'),
        CheckConstraint(status.in_(RunStatusTypes), name='status'),
        CheckConstraint(lifecycle_stage.in_(LifecycleStageTypes), name='lifecycle_stage'),
    )

    def __repr__(self):
        return '<SqlrunInfo(uuid={}, experiment_id={})'.format(self.run_uuid, self.experiment_id)


class SqlRun(Base, EntityMixin):
    __tablename__ = 'run'
    __entity__ = Run
    __properties__ = Run._properties()
    id = Column(Integer, primary_key=True)
    info_id = Column(Integer, ForeignKey('run_info.id'))
    info = relationship('SqlRunInfo', backref=backref('run', uselist=False),
                        cascade='delete')
    data_id = Column(Integer, ForeignKey('run_data.id'))
    data = relationship('SqlRunData', backref=backref('run', uselist=False),
                        cascade='delete')

    def to_mlflow_entity(self):
        self._validate()

        # run has diff parameter names in __init__ than in properties_ so we do this manually
        run_info = self.info.to_mlflow_entity()
        run_data = self.data.to_mlflow_entity()
        return self.__entity__(run_info=run_info, run_data=run_data)

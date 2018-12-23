import time
import uuid
import os
from sqlalchemy.orm import relationship, backref
from sqlalchemy import Column, Text, String, Float, ForeignKey, Integer, CheckConstraint, Boolean, PrimaryKeyConstraint
from sqlalchemy.ext.declarative import declarative_base
from mlflow.entities import Experiment, RunTag, Metric, Param, RunData, RunInfo,\
    SourceType, RunStatus, Run

Base = declarative_base()


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


def generate_uuid():
    return uuid.uuid4().hex


def _create_entity(base, model):

    # create dict of kwargs properties for entity and return the intialized entity
    config = {}
    for k in base._properties():
        # check if its mlflow entity and build it
        obj = getattr(model, k)

        # Run data contains list for metrics, params and tags
        # so obj will be a list so we need to convert those items
        if k == 'metrics':
            obj = [Metric(o.key, o.value, o.timestamp) for o in obj]

        if k == 'params':
            obj = [Param(o.key, o.value) for o in obj]

        if k == 'tags':
            obj = [RunTag(o.key, o.value) for o in obj]

        config[k] = obj
    return base(**config)


class SqlExperiment(Base):
    __tablename__ = 'experiments'
    experiment_id = Column(Integer)
    is_deleted = Column(Boolean, default=False)
    name = Column(String(256), unique=True, nullable=False)
    artifact_location = Column(Text, nullable=True)
    lifecycle_stage = Column(String(32), default=Experiment.ACTIVE_LIFECYCLE)

    __table_args__ = (
        CheckConstraint(
            lifecycle_stage.in_(ExperimentLifecycleStages), name='lifecycle_stage'),
        PrimaryKeyConstraint('experiment_id')
    )

    def __repr__(self):
        return '<SqlExperiment ({}, {})>'.format(self.experiment_id, self.name)

    def to_mlflow_entity(self):
        return _create_entity(Experiment, self)


class SqlRun(Base):
    __tablename__ = 'run'

    id = Column(Integer)
    is_deleted = Column(Boolean, default=False)
    run_uuid = Column(String(32), default=generate_uuid, unique=True, nullable=False)
    name = Column(String(256), unique=True)
    source_type = Column(Integer, default=SourceType.LOCAL)
    source_name = Column(String(256))
    entry_point_name = Column(String(256))
    user_id = Column(String(256), default=_get_user_id(), nullable=False)
    status = Column(Integer, default=RunStatus.SCHEDULED)
    start_time = Column(Integer, default=int(time.time()))
    end_time = Column(Integer, nullable=True, default=None)
    source_version = Column(String(10))
    lifecycle_stage = Column(Integer, default=RunInfo.ACTIVE_LIFECYCLE)
    artifact_uri = Column(Text, default=None)

    __table_args__ = (
        CheckConstraint(source_type.in_(SourceTypes), name='source_type'),
        CheckConstraint(status.in_(RunStatusTypes), name='status'),
        CheckConstraint(lifecycle_stage.in_(LifecycleStageTypes), name='lifecycle_stage'),
        PrimaryKeyConstraint('id')
    )

    experiment_id = Column(Integer, ForeignKey('experiments.experiment_id'))
    experiment = relationship('SqlExperiment', backref=backref('runs', cascade='all,delete'))

    def to_mlflow_entity(self):

        # run has diff parameter names in __init__ than in properties_ so we do this manually
        info = _create_entity(RunInfo, self)
        data = _create_entity(RunData, self)
        return Run(run_info=info, run_data=data)


class SqlTag(Base):
    __tablename__ = 'tag'
    key = Column(Text)
    value = Column(Text, nullable=True)
    run_uuid = Column(String(32), ForeignKey('run.run_uuid'))
    run = relationship('SqlRun', backref=backref('tags', cascade='all,delete'))

    __table_args__ = (
        PrimaryKeyConstraint('key', 'run_uuid'),
    )

    def __repr__(self):
        return '<SqlRunTag({}, {})>'.format(self.key, self.value)

    def to_mlflow_entity(self):
        return _create_entity(RunTag, self)


class SqlMetric(Base):
    __tablename__ = 'metric'
    key = Column(Text)
    value = Column(Float, nullable=False)
    timestamp = Column(Integer, default=int(time.time()))
    run_uuid = Column(String(32), ForeignKey('run.run_uuid'))
    run = relationship('SqlRun', backref=backref('metrics', cascade='all,delete'))

    __table_args__ = (
        PrimaryKeyConstraint('key', 'timestamp', 'run_uuid'),
    )

    def __repr__(self):
        return '<SqlMetric({}, {})>'.format(self.key, self.value)

    def to_mlflow_entity(self):
        return _create_entity(Metric, self)


class SqlParam(Base):
    __tablename__ = 'param'
    key = Column(Text)
    value = Column(Text, nullable=False)
    run_uuid = Column(String(32), ForeignKey('run.run_uuid'))
    run = relationship('SqlRun', backref=backref('params', cascade='all,delete'))

    __table_args__ = (
        PrimaryKeyConstraint('key', 'run_uuid'),
    )

    def __repr__(self):
        return '<SqlParam({}, {})>'.format(self.key, self.value)

    def to_mlflow_entity(self):
        return _create_entity(Param, self)

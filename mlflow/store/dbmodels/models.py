import time
from sqlalchemy.orm import relationship, backref
from sqlalchemy import (
    Column, String, Float, ForeignKey, Integer, CheckConstraint,
    BigInteger, PrimaryKeyConstraint)
from sqlalchemy.ext.declarative import declarative_base
from mlflow.entities import (
    Experiment, RunTag, Metric, Param, RunData, RunInfo,
    SourceType, RunStatus, Run)

Base = declarative_base()


ExperimentLifecycleStageTypes = [
    Experiment.ACTIVE_LIFECYCLE,
    Experiment.DELETED_LIFECYCLE
]

SourceTypes = [
    SourceType.to_string(SourceType.NOTEBOOK),
    SourceType.to_string(SourceType.JOB),
    SourceType.to_string(SourceType.LOCAL),
    SourceType.to_string(SourceType.UNKNOWN),
    SourceType.to_string(SourceType.PROJECT)
]

RunStatusTypes = [
    RunStatus.to_string(RunStatus.SCHEDULED),
    RunStatus.to_string(RunStatus.FAILED),
    RunStatus.to_string(RunStatus.FINISHED),
    RunStatus.to_string(RunStatus.RUNNING)
]

RunLifecycleStageTypes = [
    RunInfo.ACTIVE_LIFECYCLE,
    RunInfo.DELETED_LIFECYCLE
]


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

    experiment_id = Column(Integer, autoincrement=True)
    name = Column(String(256), unique=True, nullable=False)
    artifact_location = Column(String(256), nullable=True)
    lifecycle_stage = Column(String(32), default=Experiment.ACTIVE_LIFECYCLE)

    __table_args__ = (
        CheckConstraint(
            lifecycle_stage.in_(ExperimentLifecycleStageTypes), name='lifecycle_stage'),
        PrimaryKeyConstraint('experiment_id', name='experiment_pk')
    )

    def __repr__(self):
        return '<SqlExperiment ({}, {})>'.format(self.experiment_id, self.name)

    def to_mlflow_entity(self):
        return _create_entity(Experiment, self)


class SqlRun(Base):
    __tablename__ = 'runs'

    run_uuid = Column(String(32), nullable=False)
    name = Column(String(250))
    source_type = Column(String(20), default=SourceType.to_string(SourceType.LOCAL))
    source_name = Column(String(500))
    entry_point_name = Column(String(50))
    user_id = Column(String(256), nullable=True, default=None)
    status = Column(String(20), default=RunStatus.to_string(RunStatus.SCHEDULED))
    start_time = Column(BigInteger, default=int(time.time()))
    end_time = Column(BigInteger, nullable=True, default=None)
    source_version = Column(String(50))
    lifecycle_stage = Column(String(20), default=RunInfo.ACTIVE_LIFECYCLE)
    artifact_uri = Column(String(20), default=None)
    experiment_id = Column(Integer, ForeignKey('experiments.experiment_id'))
    experiment = relationship('SqlExperiment', backref=backref('runs', cascade='all'))

    __table_args__ = (
        CheckConstraint(source_type.in_(SourceTypes), name='source_type'),
        CheckConstraint(status.in_(RunStatusTypes), name='status'),
        CheckConstraint(lifecycle_stage.in_(RunLifecycleStageTypes), name='lifecycle_stage'),
        PrimaryKeyConstraint('run_uuid', name='run_pk')
    )

    def to_mlflow_entity(self):
        # run has diff parameter names in __init__ than in properties_ so we do this manually
        info = _create_entity(RunInfo, self)
        data = _create_entity(RunData, self)
        return Run(run_info=info, run_data=data)


class SqlTag(Base):
    __tablename__ = 'tags'

    key = Column(String(250))
    value = Column(String(250), nullable=True)
    run_uuid = Column(String(32), ForeignKey('runs.run_uuid'))
    run = relationship('SqlRun', backref=backref('tags', cascade='all'))

    __table_args__ = (
        PrimaryKeyConstraint('key', 'run_uuid', name='tag_pk'),
    )

    def __repr__(self):
        return '<SqlRunTag({}, {})>'.format(self.key, self.value)

    def to_mlflow_entity(self):
        return _create_entity(RunTag, self)


class SqlMetric(Base):
    __tablename__ = 'metrics'

    key = Column(String(250))
    value = Column(Float, nullable=False)
    timestamp = Column(BigInteger, default=int(time.time()))
    run_uuid = Column(String(32), ForeignKey('runs.run_uuid'))
    run = relationship('SqlRun', backref=backref('metrics', cascade='all'))

    __table_args__ = (
        PrimaryKeyConstraint('key', 'timestamp', 'run_uuid', name='metric_pk'),
    )

    def __repr__(self):
        return '<SqlMetric({}, {})>'.format(self.key, self.value)

    def to_mlflow_entity(self):
        return _create_entity(Metric, self)


class SqlParam(Base):
    __tablename__ = 'params'

    key = Column(String(250))
    value = Column(String(250), nullable=False)
    run_uuid = Column(String(32), ForeignKey('runs.run_uuid'))
    run = relationship('SqlRun', backref=backref('params', cascade='all'))

    __table_args__ = (
        PrimaryKeyConstraint('key', 'run_uuid', name='param_pk'),
    )

    def __repr__(self):
        return '<SqlParam({}, {})>'.format(self.key, self.value)

    def to_mlflow_entity(self):
        return _create_entity(Param, self)

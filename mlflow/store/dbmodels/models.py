import time
from sqlalchemy.orm import relationship, backref
from sqlalchemy import (
    Column, String, Float, ForeignKey, Integer, CheckConstraint,
    BigInteger, PrimaryKeyConstraint)
from sqlalchemy.ext.declarative import declarative_base
from mlflow.entities import (
    Experiment, RunTag, Metric, Param, RunData, RunInfo,
    SourceType, RunStatus, Run, ViewType)
from mlflow.entities.lifecycle_stage import LifecycleStage

Base = declarative_base()


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


def _create_entity(base, model):

    # create dict of kwargs properties for entity and return the initialized entity
    config = {}
    for k in base._properties():
        # check if its mlflow entity and build it
        obj = getattr(model, k)

        if isinstance(model, SqlRun):
            if base is RunData:
                # Run data contains list for metrics, params and tags
                # so obj will be a list so we need to convert those items
                if k == 'metrics':
                    # only get latest recorded metrics per key
                    metrics = {}
                    for o in obj:
                        if o.key not in metrics or o.timestamp > metrics.get(o.key).timestamp:
                            metrics[o.key] = Metric(o.key, o.value, o.timestamp)
                    obj = metrics.values()
                elif k == 'params':
                    obj = [Param(o.key, o.value) for o in obj]
                elif k == 'tags':
                    obj = [RunTag(o.key, o.value) for o in obj]
            elif base is RunInfo:
                if k == 'source_type':
                    obj = SourceType.from_string(obj)
                elif k == "status":
                    obj = RunStatus.from_string(obj)

        config[k] = obj
    return base(**config)


class SqlExperiment(Base):
    """
    DB model for :py:class:`mlflow.entities.Experiment`. These are recorded in ``experiment`` table.
    """
    __tablename__ = 'experiments'

    experiment_id = Column(Integer, autoincrement=True)
    """
    Experiment ID: `Integer`. *Primary Key* for ``experiment`` table.
    """
    name = Column(String(256), unique=True, nullable=False)
    """
    Experiment name: `String` (limit 256 characters). Defined as *Unique* and *Non null* in
                     table schema.
    """
    artifact_location = Column(String(256), nullable=True)
    """
    Default artifact location for this experiment: `String` (limit 256 characters). Defined as
                                                    *Non null* in table schema.
    """
    lifecycle_stage = Column(String(32), default=LifecycleStage.ACTIVE)
    """
    Lifecycle Stage of experiment: `String` (limit 32 characters).
                                    Can be either ``active`` (default) or ``deleted``.
    """

    __table_args__ = (
        CheckConstraint(
            lifecycle_stage.in_(LifecycleStage.view_type_to_stages(ViewType.ALL)),
            name='lifecycle_stage'),
        PrimaryKeyConstraint('experiment_id', name='experiment_pk')
    )

    def __repr__(self):
        return '<SqlExperiment ({}, {})>'.format(self.experiment_id, self.name)

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        :return: :py:class:`mlflow.entities.Experiment`.
        """
        return _create_entity(Experiment, self)


class SqlRun(Base):
    """
    DB model for :py:class:`mlflow.entities.Run`. These are recorded in ``runs`` table.
    """
    __tablename__ = 'runs'

    run_uuid = Column(String(32), nullable=False)
    """
    Run UUID: `String` (limit 32 characters). *Primary Key* for ``runs`` table.
    """
    name = Column(String(250))
    """
    Run name: `String` (limit 250 characters).
    """
    source_type = Column(String(20), default=SourceType.to_string(SourceType.LOCAL))
    """
    Source Type: `String` (limit 20 characters). Can be one of ``NOTEBOOK``, ``JOB``, ``PROJECT``,
                 ``LOCAL`` (default), or ``UNKNOWN``.
    """
    source_name = Column(String(500))
    """
    Name of source recording the run: `String` (limit 500 characters).
    """
    entry_point_name = Column(String(50))
    """
    Entry-point name that launched the run run: `String` (limit 50 characters).
    """
    user_id = Column(String(256), nullable=True, default=None)
    """
    User ID: `String` (limit 256 characters). Defaults to ``null``.
    """
    status = Column(String(20), default=RunStatus.to_string(RunStatus.SCHEDULED))
    """
    Run Status: `String` (limit 20 characters). Can be one of ``RUNNING``, ``SCHEDULED`` (default),
                ``FINISHED``, ``FAILED``.
    """
    start_time = Column(BigInteger, default=int(time.time()))
    """
    Run start time: `BigInteger`. Defaults to current system time.
    """
    end_time = Column(BigInteger, nullable=True, default=None)
    """
    Run end time: `BigInteger`.
    """
    source_version = Column(String(50))
    """
    Source version: `String` (limit 50 characters).
    """
    lifecycle_stage = Column(String(20), default=LifecycleStage.ACTIVE)
    """
    Lifecycle Stage of run: `String` (limit 32 characters).
                            Can be either ``active`` (default) or ``deleted``.
    """
    artifact_uri = Column(String(200), default=None)
    """
    Default artifact location for this run: `String` (limit 200 characters).
    """
    experiment_id = Column(Integer, ForeignKey('experiments.experiment_id'))
    """
    Experiment ID to which this run belongs to: *Foreign Key* into ``experiment`` table.
    """
    experiment = relationship('SqlExperiment', backref=backref('runs', cascade='all'))
    """
    SQLAlchemy relationship (many:one) with :py:class:`mlflow.store.dbmodels.models.SqlExperiment`.
    """

    __table_args__ = (
        CheckConstraint(source_type.in_(SourceTypes), name='source_type'),
        CheckConstraint(status.in_(RunStatusTypes), name='status'),
        CheckConstraint(lifecycle_stage.in_(LifecycleStage.view_type_to_stages(ViewType.ALL)),
                        name='lifecycle_stage'),
        PrimaryKeyConstraint('run_uuid', name='run_pk')
    )

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        :return: :py:class:`mlflow.entities.Run`.
        """
        # run has diff parameter names in __init__ than in properties_ so we do this manually
        info = _create_entity(RunInfo, self)
        data = _create_entity(RunData, self)
        return Run(run_info=info, run_data=data)


class SqlTag(Base):
    """
    DB model for :py:class:`mlflow.entities.RunTag`. These are recorded in ``tags`` table.
    """
    __tablename__ = 'tags'

    key = Column(String(250))
    """
    Tag key: `String` (limit 250 characters). *Primary Key* for ``tags`` table.
    """
    value = Column(String(250), nullable=True)
    """
    Value associated with tag: `String` (limit 250 characters). Could be *null*.
    """
    run_uuid = Column(String(32), ForeignKey('runs.run_uuid'))
    """
    Run UUID to which this tag belongs to: *Foreign Key* into ``runs`` table.
    """
    run = relationship('SqlRun', backref=backref('tags', cascade='all'))
    """
    SQLAlchemy relationship (many:one) with :py:class:`mlflow.store.dbmodels.models.SqlRun`.
    """

    __table_args__ = (
        PrimaryKeyConstraint('key', 'run_uuid', name='tag_pk'),
    )

    def __repr__(self):
        return '<SqlRunTag({}, {})>'.format(self.key, self.value)

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        :return: :py:class:`mlflow.entities.RunTag`.
        """
        return _create_entity(RunTag, self)


class SqlMetric(Base):
    __tablename__ = 'metrics'

    key = Column(String(250))
    """
    Metric key: `String` (limit 250 characters). Part of *Primary Key* for ``metrics`` table.
    """
    value = Column(Float, nullable=False)
    """
    Metric value: `Float`. Defined as *Non-null* in schema.
    """
    timestamp = Column(BigInteger, default=lambda: int(time.time()))
    """
    Timestamp recorded for this metric entry: `BigInteger`. Part of *Primary Key* for
                                               ``metrics`` table.
    """
    run_uuid = Column(String(32), ForeignKey('runs.run_uuid'))
    """
    Run UUID to which this metric belongs to: Part of *Primary Key* for ``metrics`` table.
                                              *Foreign Key* into ``runs`` table.
    """
    run = relationship('SqlRun', backref=backref('metrics', cascade='all'))
    """
    SQLAlchemy relationship (many:one) with :py:class:`mlflow.store.dbmodels.models.SqlRun`.
    """

    __table_args__ = (
        PrimaryKeyConstraint('key', 'timestamp', 'run_uuid', name='metric_pk'),
    )

    def __repr__(self):
        return '<SqlMetric({}, {}, {})>'.format(self.key, self.value, self.timestamp)

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        :return: :py:class:`mlflow.entities.Metric`.
        """
        return _create_entity(Metric, self)


class SqlParam(Base):
    __tablename__ = 'params'

    key = Column(String(250))
    """
    Param key: `String` (limit 250 characters). Part of *Primary Key* for ``params`` table.
    """
    value = Column(String(250), nullable=False)
    """
    Param value: `String` (limit 250 characters). Defined as *Non-null* in schema.
    """
    run_uuid = Column(String(32), ForeignKey('runs.run_uuid'))
    """
    Run UUID to which this metric belongs to: Part of *Primary Key* for ``params`` table.
                                              *Foreign Key* into ``runs`` table.
    """
    run = relationship('SqlRun', backref=backref('params', cascade='all'))
    """
    SQLAlchemy relationship (many:one) with :py:class:`mlflow.store.dbmodels.models.SqlRun`.
    """

    __table_args__ = (
        PrimaryKeyConstraint('key', 'run_uuid', name='param_pk'),
    )

    def __repr__(self):
        return '<SqlParam({}, {})>'.format(self.key, self.value)

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        :return: :py:class:`mlflow.entities.Param`.
        """
        return _create_entity(Param, self)

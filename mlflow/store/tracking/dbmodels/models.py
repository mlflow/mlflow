from sqlalchemy.orm import relationship, backref
import sqlalchemy as sa
from sqlalchemy import (
    Column,
    String,
    ForeignKey,
    Integer,
    CheckConstraint,
    BigInteger,
    PrimaryKeyConstraint,
    Boolean,
    Index,
)
from mlflow.entities import (
    Experiment,
    RunTag,
    Metric,
    Param,
    RunData,
    RunInfo,
    SourceType,
    RunStatus,
    Run,
    ViewType,
    ExperimentTag,
)
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.store.db.base_sql_model import Base
from mlflow.utils.mlflow_tags import _get_run_name_from_tags
from mlflow.utils.time_utils import get_current_time_millis

SourceTypes = [
    SourceType.to_string(SourceType.NOTEBOOK),
    SourceType.to_string(SourceType.JOB),
    SourceType.to_string(SourceType.LOCAL),
    SourceType.to_string(SourceType.UNKNOWN),
    SourceType.to_string(SourceType.PROJECT),
]

RunStatusTypes = [
    RunStatus.to_string(RunStatus.SCHEDULED),
    RunStatus.to_string(RunStatus.FAILED),
    RunStatus.to_string(RunStatus.FINISHED),
    RunStatus.to_string(RunStatus.RUNNING),
    RunStatus.to_string(RunStatus.KILLED),
]


class SqlExperiment(Base):
    """
    DB model for :py:class:`mlflow.entities.Experiment`. These are recorded in ``experiment`` table.
    """

    __tablename__ = "experiments"

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
    creation_time = Column(BigInteger(), default=get_current_time_millis)
    """
    Creation time of experiment: `BigInteger`.
    """
    last_update_time = Column(BigInteger(), default=get_current_time_millis)
    """
    Last Update time of experiment: `BigInteger`.
    """

    __table_args__ = (
        CheckConstraint(
            lifecycle_stage.in_(LifecycleStage.view_type_to_stages(ViewType.ALL)),
            name="experiments_lifecycle_stage",
        ),
        PrimaryKeyConstraint("experiment_id", name="experiment_pk"),
    )

    def __repr__(self):
        return "<SqlExperiment ({}, {})>".format(self.experiment_id, self.name)

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        :return: :py:class:`mlflow.entities.Experiment`.
        """
        return Experiment(
            experiment_id=str(self.experiment_id),
            name=self.name,
            artifact_location=self.artifact_location,
            lifecycle_stage=self.lifecycle_stage,
            tags=[t.to_mlflow_entity() for t in self.tags],
            creation_time=self.creation_time,
            last_update_time=self.last_update_time,
        )


class SqlRun(Base):
    """
    DB model for :py:class:`mlflow.entities.Run`. These are recorded in ``runs`` table.
    """

    __tablename__ = "runs"

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
    start_time = Column(BigInteger, default=get_current_time_millis)
    """
    Run start time: `BigInteger`. Defaults to current system time.
    """
    end_time = Column(BigInteger, nullable=True, default=None)
    """
    Run end time: `BigInteger`.
    """
    deleted_time = Column(BigInteger, nullable=True, default=None)
    """
    Run deleted time: `BigInteger`. Timestamp of when run is deleted, defaults to none.
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
    experiment_id = Column(Integer, ForeignKey("experiments.experiment_id"))
    """
    Experiment ID to which this run belongs to: *Foreign Key* into ``experiment`` table.
    """
    experiment = relationship("SqlExperiment", backref=backref("runs", cascade="all"))
    """
    SQLAlchemy relationship (many:one) with :py:class:`mlflow.store.dbmodels.models.SqlExperiment`.
    """

    __table_args__ = (
        CheckConstraint(source_type.in_(SourceTypes), name="source_type"),
        CheckConstraint(status.in_(RunStatusTypes), name="status"),
        CheckConstraint(
            lifecycle_stage.in_(LifecycleStage.view_type_to_stages(ViewType.ALL)),
            name="runs_lifecycle_stage",
        ),
        PrimaryKeyConstraint("run_uuid", name="run_pk"),
    )

    @staticmethod
    def get_attribute_name(mlflow_attribute_name):
        """
        Resolves an MLflow attribute name to a `SqlRun` attribute name.
        """
        # Currently, MLflow Search attributes defined in `SearchUtils.VALID_SEARCH_ATTRIBUTE_KEYS`
        # share the same names as their corresponding `SqlRun` attributes. Therefore, this function
        # returns the same attribute name
        return {"run_name": "name", "run_id": "run_uuid"}.get(
            mlflow_attribute_name, mlflow_attribute_name
        )

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        :return: :py:class:`mlflow.entities.Run`.
        """
        run_info = RunInfo(
            run_uuid=self.run_uuid,
            run_id=self.run_uuid,
            run_name=self.name,
            experiment_id=str(self.experiment_id),
            user_id=self.user_id,
            status=self.status,
            start_time=self.start_time,
            end_time=self.end_time,
            lifecycle_stage=self.lifecycle_stage,
            artifact_uri=self.artifact_uri,
        )

        tags = [t.to_mlflow_entity() for t in self.tags]
        run_data = RunData(
            metrics=[m.to_mlflow_entity() for m in self.latest_metrics],
            params=[p.to_mlflow_entity() for p in self.params],
            tags=tags,
        )
        if not run_info.run_name:
            run_name = _get_run_name_from_tags(tags)
            if run_name:
                run_info._set_run_name(run_name)

        return Run(run_info=run_info, run_data=run_data)


class SqlExperimentTag(Base):
    """
    DB model for :py:class:`mlflow.entities.RunTag`.
    These are recorded in ``experiment_tags`` table.
    """

    __tablename__ = "experiment_tags"

    key = Column(String(250))
    """
    Tag key: `String` (limit 250 characters). *Primary Key* for ``tags`` table.
    """
    value = Column(String(5000), nullable=True)
    """
    Value associated with tag: `String` (limit 5000 characters). Could be *null*.
    """
    experiment_id = Column(Integer, ForeignKey("experiments.experiment_id"))
    """
    Experiment ID to which this tag belongs: *Foreign Key* into ``experiments`` table.
    """
    experiment = relationship("SqlExperiment", backref=backref("tags", cascade="all"))
    """
    SQLAlchemy relationship (many:one) with :py:class:`mlflow.store.dbmodels.models.SqlExperiment`.
    """

    __table_args__ = (PrimaryKeyConstraint("key", "experiment_id", name="experiment_tag_pk"),)

    def __repr__(self):
        return "<SqlExperimentTag({}, {})>".format(self.key, self.value)

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        :return: :py:class:`mlflow.entities.RunTag`.
        """
        return ExperimentTag(key=self.key, value=self.value)


class SqlTag(Base):
    """
    DB model for :py:class:`mlflow.entities.RunTag`. These are recorded in ``tags`` table.
    """

    __tablename__ = "tags"
    __table_args__ = (
        PrimaryKeyConstraint("key", "run_uuid", name="tag_pk"),
        Index(f"index_{__tablename__}_run_uuid", "run_uuid"),
    )

    key = Column(String(250))
    """
    Tag key: `String` (limit 250 characters). *Primary Key* for ``tags`` table.
    """
    value = Column(String(5000), nullable=True)
    """
    Value associated with tag: `String` (limit 250 characters). Could be *null*.
    """
    run_uuid = Column(String(32), ForeignKey("runs.run_uuid"))
    """
    Run UUID to which this tag belongs to: *Foreign Key* into ``runs`` table.
    """
    run = relationship("SqlRun", backref=backref("tags", cascade="all"))
    """
    SQLAlchemy relationship (many:one) with :py:class:`mlflow.store.dbmodels.models.SqlRun`.
    """

    def __repr__(self):
        return "<SqlRunTag({}, {})>".format(self.key, self.value)

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        :return: :py:class:`mlflow.entities.RunTag`.
        """
        return RunTag(key=self.key, value=self.value)


class SqlMetric(Base):
    __tablename__ = "metrics"
    __table_args__ = (
        PrimaryKeyConstraint(
            "key", "timestamp", "step", "run_uuid", "value", "is_nan", name="metric_pk"
        ),
        Index(f"index_{__tablename__}_run_uuid", "run_uuid"),
    )

    key = Column(String(250))
    """
    Metric key: `String` (limit 250 characters). Part of *Primary Key* for ``metrics`` table.
    """
    value = Column(sa.types.Float(precision=53), nullable=False)
    """
    Metric value: `Float`. Defined as *Non-null* in schema.
    """
    timestamp = Column(BigInteger, default=get_current_time_millis)
    """
    Timestamp recorded for this metric entry: `BigInteger`. Part of *Primary Key* for
                                               ``metrics`` table.
    """
    step = Column(BigInteger, default=0, nullable=False)
    """
    Step recorded for this metric entry: `BigInteger`.
    """
    is_nan = Column(Boolean(create_constraint=True), nullable=False, default=False)
    """
    True if the value is in fact NaN.
    """
    run_uuid = Column(String(32), ForeignKey("runs.run_uuid"))
    """
    Run UUID to which this metric belongs to: Part of *Primary Key* for ``metrics`` table.
                                              *Foreign Key* into ``runs`` table.
    """
    run = relationship("SqlRun", backref=backref("metrics", cascade="all"))
    """
    SQLAlchemy relationship (many:one) with :py:class:`mlflow.store.dbmodels.models.SqlRun`.
    """

    def __repr__(self):
        return "<SqlMetric({}, {}, {}, {})>".format(self.key, self.value, self.timestamp, self.step)

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        :return: :py:class:`mlflow.entities.Metric`.
        """
        return Metric(
            key=self.key,
            value=self.value if not self.is_nan else float("nan"),
            timestamp=self.timestamp,
            step=self.step,
        )


class SqlLatestMetric(Base):
    __tablename__ = "latest_metrics"
    __table_args__ = (
        PrimaryKeyConstraint("key", "run_uuid", name="latest_metric_pk"),
        Index(f"index_{__tablename__}_run_uuid", "run_uuid"),
    )

    key = Column(String(250))
    """
    Metric key: `String` (limit 250 characters). Part of *Primary Key* for ``latest_metrics`` table.
    """
    value = Column(sa.types.Float(precision=53), nullable=False)
    """
    Metric value: `Float`. Defined as *Non-null* in schema.
    """
    timestamp = Column(BigInteger, default=get_current_time_millis)
    """
    Timestamp recorded for this metric entry: `BigInteger`. Part of *Primary Key* for
                                               ``latest_metrics`` table.
    """
    step = Column(BigInteger, default=0, nullable=False)
    """
    Step recorded for this metric entry: `BigInteger`.
    """
    is_nan = Column(Boolean(create_constraint=True), nullable=False, default=False)
    """
    True if the value is in fact NaN.
    """
    run_uuid = Column(String(32), ForeignKey("runs.run_uuid"))
    """
    Run UUID to which this metric belongs to: Part of *Primary Key* for ``latest_metrics`` table.
                                              *Foreign Key* into ``runs`` table.
    """
    run = relationship("SqlRun", backref=backref("latest_metrics", cascade="all"))
    """
    SQLAlchemy relationship (many:one) with :py:class:`mlflow.store.dbmodels.models.SqlRun`.
    """

    def __repr__(self):
        return "<SqlLatestMetric({}, {}, {}, {})>".format(
            self.key, self.value, self.timestamp, self.step
        )

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        :return: :py:class:`mlflow.entities.Metric`.
        """
        return Metric(
            key=self.key,
            value=self.value if not self.is_nan else float("nan"),
            timestamp=self.timestamp,
            step=self.step,
        )


class SqlParam(Base):
    __tablename__ = "params"
    __table_args__ = (
        PrimaryKeyConstraint("key", "run_uuid", name="param_pk"),
        Index(f"index_{__tablename__}_run_uuid", "run_uuid"),
    )

    key = Column(String(250))
    """
    Param key: `String` (limit 250 characters). Part of *Primary Key* for ``params`` table.
    """
    value = Column(String(500), nullable=False)
    """
    Param value: `String` (limit 500 characters). Defined as *Non-null* in schema.
    """
    run_uuid = Column(String(32), ForeignKey("runs.run_uuid"))
    """
    Run UUID to which this metric belongs to: Part of *Primary Key* for ``params`` table.
                                              *Foreign Key* into ``runs`` table.
    """
    run = relationship("SqlRun", backref=backref("params", cascade="all"))
    """
    SQLAlchemy relationship (many:one) with :py:class:`mlflow.store.dbmodels.models.SqlRun`.
    """

    def __repr__(self):
        return "<SqlParam({}, {})>".format(self.key, self.value)

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        :return: :py:class:`mlflow.entities.Param`.
        """
        return Param(key=self.key, value=self.value)

import json

import sqlalchemy as sa
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Computed,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    UnicodeText,
)
from sqlalchemy.orm import backref, relationship

from mlflow.entities import (
    Assessment,
    AssessmentError,
    AssessmentSource,
    Dataset,
    Expectation,
    Experiment,
    ExperimentTag,
    Feedback,
    InputTag,
    Metric,
    Param,
    Run,
    RunData,
    RunInfo,
    RunStatus,
    RunTag,
    SourceType,
    TraceInfo,
    ViewType,
)
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities.logged_model import LoggedModel
from mlflow.entities.logged_model_parameter import LoggedModelParameter
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.entities.logged_model_tag import LoggedModelTag
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.store.db.base_sql_model import Base
from mlflow.tracing.utils import generate_assessment_id
from mlflow.utils.mlflow_tags import _get_run_name_from_tags
from mlflow.utils.time import get_current_time_millis

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
        return f"<SqlExperiment ({self.experiment_id}, {self.name})>"

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        Returns:
            :py:class:`mlflow.entities.Experiment`.
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

        Returns:
            mlflow.entities.Run: Description of the return value.
        """
        run_info = RunInfo(
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
        return f"<SqlExperimentTag({self.key}, {self.value})>"

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        Returns:
            mlflow.entities.RunTag: Description of the return value.
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
    value = Column(String(8000), nullable=True)
    """
    Value associated with tag: `String` (limit 8000 characters). Could be *null*.
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
        return f"<SqlRunTag({self.key}, {self.value})>"

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        Returns:
            :py:class:`mlflow.entities.RunTag`.
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
        return f"<SqlMetric({self.key}, {self.value}, {self.timestamp}, {self.step})>"

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        Returns:
            mlflow.entities.Metric: Description of the return value.
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
        return f"<SqlLatestMetric({self.key}, {self.value}, {self.timestamp}, {self.step})>"

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        Returns:
            mlflow.entities.Metric: Description of the return value.
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
    value = Column(String(8000), nullable=False)
    """
    Param value: `String` (limit 8000 characters). Defined as *Non-null* in schema.
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
        return f"<SqlParam({self.key}, {self.value})>"

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        Returns:
            mlflow.entities.Param: Description of the return value.
        """
        return Param(key=self.key, value=self.value)


class SqlDataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (
        PrimaryKeyConstraint("experiment_id", "name", "digest", name="dataset_pk"),
        Index(f"index_{__tablename__}_dataset_uuid", "dataset_uuid"),
        Index(
            f"index_{__tablename__}_experiment_id_dataset_source_type",
            "experiment_id",
            "dataset_source_type",
        ),
    )

    dataset_uuid = Column(String(36), nullable=False)
    """
    Dataset UUID: `String` (limit 36 characters). Defined as *Non-null* in schema.
    Part of *Primary Key* for ``datasets`` table.
    """
    experiment_id = Column(Integer, ForeignKey("experiments.experiment_id", ondelete="CASCADE"))
    """
    Experiment ID to which this dataset belongs: *Foreign Key* into ``experiments`` table.
    """
    name = Column(String(500), nullable=False)
    """
    Param name: `String` (limit 500 characters). Defined as *Non-null* in schema.
    Part of *Primary Key* for ``datasets`` table.
    """
    digest = Column(String(36), nullable=False)
    """
    Param digest: `String` (limit 500 characters). Defined as *Non-null* in schema.
    Part of *Primary Key* for ``datasets`` table.
    """
    dataset_source_type = Column(String(36), nullable=False)
    """
    Param dataset_source_type: `String` (limit 36 characters). Defined as *Non-null* in schema.
    """
    dataset_source = Column(UnicodeText, nullable=False)
    """
    Param dataset_source: `UnicodeText`. Defined as *Non-null* in schema.
    """
    dataset_schema = Column(UnicodeText, nullable=True)
    """
    Param dataset_schema: `UnicodeText`.
    """
    dataset_profile = Column(UnicodeText, nullable=True)
    """
    Param dataset_profile: `UnicodeText`.
    """

    def __repr__(self):
        return "<SqlDataset ({}, {}, {}, {}, {}, {}, {}, {})>".format(
            self.dataset_uuid,
            self.experiment_id,
            self.name,
            self.digest,
            self.dataset_source_type,
            self.dataset_source,
            self.dataset_schema,
            self.dataset_profile,
        )

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        Returns:
            mlflow.entities.Dataset.
        """
        return Dataset(
            name=self.name,
            digest=self.digest,
            source_type=self.dataset_source_type,
            source=self.dataset_source,
            schema=self.dataset_schema,
            profile=self.dataset_profile,
        )


class SqlInput(Base):
    __tablename__ = "inputs"
    __table_args__ = (
        PrimaryKeyConstraint(
            "source_type", "source_id", "destination_type", "destination_id", name="inputs_pk"
        ),
        Index(f"index_{__tablename__}_input_uuid", "input_uuid"),
        Index(
            f"index_{__tablename__}_destination_type_destination_id_source_type",
            "destination_type",
            "destination_id",
            "source_type",
        ),
    )

    input_uuid = Column(String(36), nullable=False)
    """
    Input UUID: `String` (limit 36 characters). Defined as *Non-null* in schema.
    """
    source_type = Column(String(36), nullable=False)
    """
    Source type: `String` (limit 36 characters). Defined as *Non-null* in schema.
    Part of *Primary Key* for ``inputs`` table.
    """
    source_id = Column(String(36), nullable=False)
    """
    Source Id: `String` (limit 36 characters). Defined as *Non-null* in schema.
    Part of *Primary Key* for ``inputs`` table.
    """
    destination_type = Column(String(36), nullable=False)
    """
    Destination type: `String` (limit 36 characters). Defined as *Non-null* in schema.
    Part of *Primary Key* for ``inputs`` table.
    """
    destination_id = Column(String(36), nullable=False)
    """
    Destination Id: `String` (limit 36 characters). Defined as *Non-null* in schema.
    Part of *Primary Key* for ``inputs`` table.
    """
    step = Column(BigInteger, nullable=False, server_default="0")

    def __repr__(self):
        return "<SqlInput ({}, {}, {}, {}, {})>".format(
            self.input_uuid,
            self.source_type,
            self.source_id,
            self.destination_type,
            self.destination_id,
        )


class SqlInputTag(Base):
    __tablename__ = "input_tags"
    __table_args__ = (PrimaryKeyConstraint("input_uuid", "name", name="input_tags_pk"),)

    input_uuid = Column(String(36), ForeignKey("inputs.input_uuid"), nullable=False)
    """
    Input UUID: `String` (limit 36 characters). Defined as *Non-null* in schema.
    *Foreign Key* into ``inputs`` table. Part of *Primary Key* for ``input_tags`` table.
    """
    name = Column(String(255), nullable=False)
    """
    Param name: `String` (limit 255 characters). Defined as *Non-null* in schema.
    Part of *Primary Key* for ``input_tags`` table.
    """
    value = Column(String(500), nullable=False)
    """
    Param value: `String` (limit 500 characters). Defined as *Non-null* in schema.
    Part of *Primary Key* for ``input_tags`` table.
    """

    def __repr__(self):
        return f"<SqlInputTag ({self.input_uuid}, {self.name}, {self.value})>"

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        Returns:
            mlflow.entities.InputTag: Description of the return value.
        """
        return InputTag(key=self.name, value=self.value)


#######################################################################################
# Below are Tracing models. We may refactor them to be in a separate module in the future.
#######################################################################################


class SqlTraceInfo(Base):
    __tablename__ = "trace_info"

    request_id = Column(String(50), nullable=False)
    """
    Trace ID: `String` (limit 50 characters). *Primary Key* for ``trace_info`` table.
    Named as "trace_id" in V3 format.
    """
    experiment_id = Column(Integer, ForeignKey("experiments.experiment_id"), nullable=False)
    """
    Experiment ID to which this trace belongs: *Foreign Key* into ``experiments`` table.
    """
    timestamp_ms = Column(BigInteger, nullable=False)
    """
    Start time of the trace, in milliseconds. Named as "request_time" in V3 format.
    """
    execution_time_ms = Column(BigInteger, nullable=True)
    """
    Duration of the trace, in milliseconds. Could be *null* if the trace is still in progress
    or not ended correctly for some reason. Named as "execution_duration" in V3 format.
    """
    status = Column(String(50), nullable=False)
    """
    State of the trace. The values are defined in
    :py:class:`mlflow.entities.trace_status.TraceStatus` enum but we don't enforce
    constraint at DB level. Named as "state" in V3 format.
    """
    client_request_id = Column(String(50), nullable=True)
    """
    Client request ID: `String` (limit 50 characters). Could be *null*. Newly added in V3 format.
    """
    request_preview = Column(String(1000), nullable=True)
    """
    Request preview: `String` (limit 1000 characters). Could be *null*. Newly added in V3 format.
    """
    response_preview = Column(String(1000), nullable=True)
    """
    Response preview: `String` (limit 1000 characters). Could be *null*. Newly added in V3 format.
    """

    __table_args__ = (
        PrimaryKeyConstraint("request_id", name="trace_info_pk"),
        # The most frequent query will be get all traces in an experiment sorted by timestamp desc,
        # which is the default view in the UI. Also every search query should have experiment_id(s)
        # in the where clause.
        Index(f"index_{__tablename__}_experiment_id_timestamp_ms", "experiment_id", "timestamp_ms"),
    )

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        Returns:
            :py:class:`mlflow.entities.TraceInfo` object.
        """
        return TraceInfo(
            trace_id=self.request_id,
            trace_location=TraceLocation.from_experiment_id(str(self.experiment_id)),
            request_time=self.timestamp_ms,
            execution_duration=self.execution_time_ms,
            state=TraceState(self.status),
            tags={t.key: t.value for t in self.tags},
            trace_metadata={m.key: m.value for m in self.request_metadata},
            client_request_id=self.client_request_id,
            request_preview=self.request_preview,
            response_preview=self.response_preview,
            assessments=[a.to_mlflow_entity() for a in self.assessments],
        )


class SqlTraceTag(Base):
    __tablename__ = "trace_tags"

    key = Column(String(250))
    """
    Tag key: `String` (limit 250 characters).
    """
    value = Column(String(8000), nullable=True)
    """
    Value associated with tag: `String` (limit 250 characters). Could be *null*.
    """
    request_id = Column(
        String(50), ForeignKey("trace_info.request_id", ondelete="CASCADE"), nullable=False
    )
    """
    Request ID to which this tag belongs: *Foreign Key* into ``trace_info`` table.
    """
    trace_info = relationship("SqlTraceInfo", backref=backref("tags", cascade="all"))
    """
    SQLAlchemy relationship (many:one) with
    :py:class:`mlflow.store.dbmodels.models.SqlTraceInfo`.
    """

    # Key is unique within a request_id
    __table_args__ = (
        PrimaryKeyConstraint("request_id", "key", name="trace_tag_pk"),
        Index(f"index_{__tablename__}_request_id"),
    )


class SqlTraceMetadata(Base):
    __tablename__ = "trace_request_metadata"

    key = Column(String(250))
    """
    Metadata key: `String` (limit 250 characters).
    """
    value = Column(String(8000), nullable=True)
    """
    Value associated with metadata: `String` (limit 250 characters). Could be *null*.
    """
    request_id = Column(
        String(50), ForeignKey("trace_info.request_id", ondelete="CASCADE"), nullable=False
    )
    """
    Request ID to which this metadata belongs: *Foreign Key* into ``trace_info`` table.
    **Corresponding to the "trace_id" in V3 format.**
    """
    trace_info = relationship("SqlTraceInfo", backref=backref("request_metadata", cascade="all"))
    """
    SQLAlchemy relationship (many:one) with
    :py:class:`mlflow.store.dbmodels.models.SqlTraceInfo`.
    """

    # Key is unique within a request_id
    __table_args__ = (
        PrimaryKeyConstraint("request_id", "key", name="trace_request_metadata_pk"),
        Index(f"index_{__tablename__}_request_id"),
    )


class SqlAssessments(Base):
    __tablename__ = "assessments"

    assessment_id = Column(String(50), nullable=False)
    """
    Assessment ID: `String` (limit 50 characters). *Primary Key* for ``assessments`` table.
    """
    trace_id = Column(
        String(50), ForeignKey("trace_info.request_id", ondelete="CASCADE"), nullable=False
    )
    """
    Trace ID that a given assessment belongs to. *Foreign Key* into ``trace_info`` table.
    """
    name = Column(String(250), nullable=False)
    """
    Assessment Name: `String` (limit of 250 characters).
    """
    assessment_type = Column(String(50), nullable=False)
    """
    Assessment type: `String` (limit 50 characters). Either "feedback" or "expectation".
    """
    value = Column(Text, nullable=False)
    """
    The assessment's value data stored as JSON: `Text` for the actual value content.
    """
    error = Column(Text, nullable=True)
    """
    AssessmentError stored as JSON: `Text` for error information (feedback only).
    """
    created_timestamp = Column(BigInteger, nullable=False)
    """
    The assessment's creation timestamp: `BigInteger`.
    """
    last_updated_timestamp = Column(BigInteger, nullable=False)
    """
    The update time of an assessment if the assessment has been updated: `BigInteger`.
    """
    source_type = Column(String(50), nullable=False)
    """
    Assessment source type: `String` (limit 50 characters). e.g., "HUMAN", "CODE", "LLM_JUDGE".
    """
    source_id = Column(String(250), nullable=True)
    """
    Assessment source ID: `String` (limit 250 characters). e.g., "evaluator@company.com".
    """
    run_id = Column(String(32), nullable=True)
    """
    Run ID associated with the assessment if generated due to a run event:
    `String` (limit of 32 characters).
    """
    span_id = Column(String(50), nullable=True)
    """
    Span ID if the assessment is applied to a Span within a Trace:
    `String` (limit of 50 characters).
    """
    rationale = Column(Text, nullable=True)
    """
    Justification for the assessment: `Text` for longer explanations.
    """
    overrides = Column(String(50), nullable=True)
    """
    Overridden assessment_id if an assessment is intended to update and replace an existing
    assessment: `String` (limit of 50 characters).
    """
    valid = Column(Boolean, nullable=False, default=True)
    """
    Indicator for whether an assessment has been marked as invalid: `Boolean`. Defaults to True.
    """
    assessment_metadata = Column(Text, nullable=True)
    """
    Assessment metadata stored as JSON: `Text` for complex metadata structures.
    """

    trace_info = relationship("SqlTraceInfo", backref=backref("assessments", cascade="all"))
    """
    SQLAlchemy relationship (many:one) with
    :py:class:`mlflow.store.dbmodels.models.SqlTraceInfo`.
    """

    __table_args__ = (
        PrimaryKeyConstraint("assessment_id", name="assessments_pkey"),
        Index(f"index_{__tablename__}_trace_id_created_timestamp", "trace_id", "created_timestamp"),
        Index(f"index_{__tablename__}_run_id_created_timestamp", "run_id", "created_timestamp"),
        Index(f"index_{__tablename__}_last_updated_timestamp", "last_updated_timestamp"),
        Index(f"index_{__tablename__}_assessment_type", "assessment_type"),
    )

    def to_mlflow_entity(self) -> Assessment:
        """Convert SqlAssessments to Assessment object."""
        value_str = self.value
        error_str = self.error
        assessment_metadata_str = self.assessment_metadata
        assessment_type_value = self.assessment_type

        parsed_value = json.loads(value_str)
        parsed_error = None
        if error_str is not None:
            error_dict = json.loads(error_str)
            parsed_error = AssessmentError.from_dictionary(error_dict)

        parsed_metadata = None
        if assessment_metadata_str is not None:
            parsed_metadata = json.loads(assessment_metadata_str)

        source = AssessmentSource(source_type=self.source_type, source_id=self.source_id)

        if assessment_type_value == "feedback":
            assessment = Feedback(
                name=self.name,
                value=parsed_value,
                error=parsed_error,
                source=source,
                trace_id=self.trace_id,
                rationale=self.rationale,
                metadata=parsed_metadata,
                span_id=self.span_id,
                create_time_ms=self.created_timestamp,
                last_update_time_ms=self.last_updated_timestamp,
                overrides=self.overrides,
                valid=self.valid,
            )
        elif assessment_type_value == "expectation":
            assessment = Expectation(
                name=self.name,
                value=parsed_value,
                source=source,
                trace_id=self.trace_id,
                metadata=parsed_metadata,
                span_id=self.span_id,
                create_time_ms=self.created_timestamp,
                last_update_time_ms=self.last_updated_timestamp,
            )
            assessment.overrides = self.overrides
            assessment.valid = self.valid
        else:
            raise ValueError(f"Unknown assessment type: {assessment_type_value}")

        assessment.run_id = self.run_id
        assessment.assessment_id = self.assessment_id

        return assessment

    @classmethod
    def from_mlflow_entity(cls, assessment: Assessment):
        if assessment.assessment_id is None:
            assessment.assessment_id = generate_assessment_id()

        current_timestamp = get_current_time_millis()

        if assessment.feedback is not None:
            assessment_type = "feedback"
            value_json = json.dumps(assessment.feedback.value)
            error_json = (
                json.dumps(assessment.feedback.error.to_dictionary())
                if assessment.feedback.error
                else None
            )
        elif assessment.expectation is not None:
            assessment_type = "expectation"
            value_json = json.dumps(assessment.expectation.value)
            error_json = None
        else:
            raise MlflowException.invalid_parameter_value(
                "Assessment must have either feedback or expectation value"
            )

        metadata_json = json.dumps(assessment.metadata) if assessment.metadata else None

        return SqlAssessments(
            assessment_id=assessment.assessment_id,
            trace_id=assessment.trace_id,
            name=assessment.name,
            assessment_type=assessment_type,
            value=value_json,
            error=error_json,
            created_timestamp=assessment.create_time_ms or current_timestamp,
            last_updated_timestamp=assessment.last_update_time_ms or current_timestamp,
            source_type=assessment.source.source_type,
            source_id=assessment.source.source_id,
            run_id=assessment.run_id,
            span_id=assessment.span_id,
            rationale=assessment.rationale,
            overrides=assessment.overrides,
            valid=True,
            assessment_metadata=metadata_json,
        )

    def __repr__(self):
        return f"<SqlAssessments({self.assessment_id}, {self.name}, {self.assessment_type})>"


class SqlLoggedModel(Base):
    __tablename__ = "logged_models"

    model_id = Column(String(36), nullable=False)
    """
    Model ID: `String` (limit 36 characters). *Primary Key* for ``logged_models`` table.
    """

    experiment_id = Column(Integer, nullable=False)
    """
    Experiment ID to which this model belongs: *Foreign Key* into ``experiments`` table.
    """

    name = Column(String(500), nullable=False)
    """
    Model name: `String` (limit 500 characters).
    """

    artifact_location = Column(String(1000), nullable=False)
    """
    Artifact location: `String` (limit 1000 characters).
    """

    creation_timestamp_ms = Column(BigInteger, nullable=False)
    """
    Creation timestamp: `BigInteger`.
    """

    last_updated_timestamp_ms = Column(BigInteger, nullable=False)
    """
    Last updated timestamp: `BigInteger`.
    """

    status = Column(Integer, nullable=False)
    """
    Status: `Integer`.
    """

    lifecycle_stage = Column(String(32), default=LifecycleStage.ACTIVE)
    """
    Lifecycle Stage of model: `String` (limit 32 characters).
    """

    model_type = Column(String(500), nullable=True)
    """
    Model type: `String` (limit 500 characters).
    """

    source_run_id = Column(String(32), nullable=True)
    """
    Source run ID: `String` (limit 32 characters).
    """

    status_message = Column(String(1000), nullable=True)
    """
    Status message: `String` (limit 1000 characters).
    """

    tags = relationship("SqlLoggedModelTag", backref="logged_model", cascade="all")
    params = relationship("SqlLoggedModelParam", backref="logged_model", cascade="all")
    metrics = relationship("SqlLoggedModelMetric", backref="logged_model", cascade="all")

    __table_args__ = (
        PrimaryKeyConstraint("model_id", name="logged_models_pk"),
        CheckConstraint(
            lifecycle_stage.in_(LifecycleStage.view_type_to_stages(ViewType.ALL)),
            name="logged_models_lifecycle_stage_check",
        ),
        ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            ondelete="CASCADE",
            name="fk_logged_models_experiment_id",
        ),
    )

    def to_mlflow_entity(self) -> LoggedModel:
        return LoggedModel(
            model_id=self.model_id,
            experiment_id=str(self.experiment_id),
            name=self.name,
            artifact_location=self.artifact_location,
            creation_timestamp=self.creation_timestamp_ms,
            last_updated_timestamp=self.last_updated_timestamp_ms,
            status=LoggedModelStatus.from_int(self.status),
            model_type=self.model_type,
            source_run_id=self.source_run_id,
            status_message=self.status_message,
            tags={t.tag_key: t.tag_value for t in self.tags} if self.tags else None,
            params={p.param_key: p.param_value for p in self.params} if self.params else None,
            metrics=[m.to_mlflow_entity() for m in self.metrics] if self.metrics else None,
        )

    ALIASES = {
        "creation_time": "creation_timestamp_ms",
        "creation_timestamp": "creation_timestamp_ms",
        "last_updated_timestamp": "last_updated_timestamp_ms",
    }

    @staticmethod
    def is_numeric(s: str) -> bool:
        return SqlLoggedModel.ALIASES.get(s, s) in {
            "creation_timestamp_ms",
            "last_updated_timestamp_ms",
        }


class SqlLoggedModelMetric(Base):
    __tablename__ = "logged_model_metrics"

    model_id = Column(String(36), nullable=False)
    """
    Model ID: `String` (limit 36 characters).
    """

    metric_name = Column(String(500), nullable=False)
    """
    Metric name: `String` (limit 500 characters).
    """

    metric_timestamp_ms = Column(BigInteger, nullable=False)
    """
    Metric timestamp: `BigInteger`.
    """

    metric_step = Column(BigInteger, nullable=False)
    """
    Metric step: `BigInteger`.
    """

    metric_value = Column(sa.types.Float(precision=53), nullable=True)
    """
    Metric value: `Float`.
    """

    experiment_id = Column(Integer, nullable=False)
    """
    Experiment ID: `Integer`.
    """

    run_id = Column(String(32), nullable=False)
    """
    Run ID: `String` (limit 32 characters).
    """

    dataset_uuid = Column(String(36), nullable=True)
    """
    Dataset UUID: `String` (limit 36 characters).
    """

    dataset_name = Column(String(500), nullable=True)
    """
    Dataset name: `String` (limit 500 characters).
    """

    dataset_digest = Column(String(36), nullable=True)
    """
    Dataset digest: `String` (limit 36 characters).
    """

    __table_args__ = (
        PrimaryKeyConstraint(
            "model_id",
            "metric_name",
            "metric_timestamp_ms",
            "metric_step",
            "run_id",
            name="logged_model_metrics_pk",
        ),
        ForeignKeyConstraint(
            ["model_id"],
            ["logged_models.model_id"],
            ondelete="CASCADE",
            name="fk_logged_model_metrics_model_id",
        ),
        ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_logged_model_metrics_experiment_id",
        ),
        ForeignKeyConstraint(
            ["run_id"],
            ["runs.run_uuid"],
            ondelete="CASCADE",
            name="fk_logged_model_metrics_run_id",
        ),
        Index("index_logged_model_metrics_model_id", "model_id"),
    )

    def to_mlflow_entity(self) -> Metric:
        return Metric(
            key=self.metric_name,
            value=self.metric_value,
            timestamp=self.metric_timestamp_ms,
            step=self.metric_step,
            run_id=self.run_id,
            dataset_name=self.dataset_name,
            dataset_digest=self.dataset_digest,
            model_id=self.model_id,
        )


class SqlLoggedModelParam(Base):
    __tablename__ = "logged_model_params"

    model_id = Column(String(36), nullable=False)
    """
    Model ID: `String` (limit 36 characters).
    """

    experiment_id = Column(Integer, nullable=False)
    """
    Experiment ID: `Integer`.
    """

    param_key = Column(String(255), nullable=False)
    """
    Param key: `String` (limit 255 characters).
    """

    param_value = Column(Text(), nullable=False)
    """
    Param value: `Text`.
    """

    __table_args__ = (
        PrimaryKeyConstraint(
            "model_id",
            "param_key",
            name="logged_model_params_pk",
        ),
        ForeignKeyConstraint(
            ["model_id"],
            ["logged_models.model_id"],
            name="fk_logged_model_params_model_id",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_logged_model_params_experiment_id",
        ),
    )

    def to_mlflow_entity(self) -> LoggedModelParameter:
        return LoggedModelParameter(key=self.param_key, value=self.param_value)


class SqlLoggedModelTag(Base):
    __tablename__ = "logged_model_tags"

    model_id = Column(String(36), nullable=False)
    """
    Model ID: `String` (limit 36 characters).
    """

    experiment_id = Column(Integer, nullable=False)
    """
    Experiment ID: `Integer`.
    """

    tag_key = Column(String(255), nullable=False)
    """
    Tag key: `String` (limit 255 characters).
    """

    tag_value = Column(Text(), nullable=False)
    """
    Tag value: `Text`.
    """

    __table_args__ = (
        PrimaryKeyConstraint(
            "model_id",
            "tag_key",
            name="logged_model_tags_pk",
        ),
        ForeignKeyConstraint(
            ["model_id"],
            ["logged_models.model_id"],
            name="fk_logged_model_tags_model_id",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_logged_model_tags_experiment_id",
        ),
    )

    def to_mlflow_entity(self) -> LoggedModelTag:
        return LoggedModelTag(key=self.tag_key, value=self.tag_value)


class SqlSpan(Base):
    __tablename__ = "spans"

    trace_id = Column(
        String(50), ForeignKey("trace_info.request_id", ondelete="CASCADE"), nullable=False
    )
    """
    Trace ID: `String` (limit 50 characters). Part of composite primary key.
    Foreign key to trace_info table.
    """

    experiment_id = Column(Integer, ForeignKey("experiments.experiment_id"), nullable=False)
    """
    Experiment ID: `Integer`. Foreign key to experiments table.
    """

    span_id = Column(String(50), nullable=False)
    """
    Span ID: `String` (limit 50 characters). Part of composite primary key.
    """

    parent_span_id = Column(String(50), nullable=True)
    """
    Parent span ID: `String` (limit 50 characters). Can be null for root spans.
    """

    name = Column(Text, nullable=True)
    """
    Span name: `Text`. Can be null.
    """

    type = Column(String(500), nullable=True)
    """
    Span type: `String` (limit 500 characters). Can be null.
    Uses String instead of Text to support MSSQL indexes.
    Limited to 500 chars to stay within MySQL's max index key length.
    """

    status = Column(String(50), nullable=False)
    """
    Span status: `String` (limit 50 characters).
    """

    start_time_unix_nano = Column(BigInteger, nullable=False)
    """
    Start time in nanoseconds since Unix epoch: `BigInteger`.
    """

    end_time_unix_nano = Column(BigInteger, nullable=True)
    """
    End time in nanoseconds since Unix epoch: `BigInteger`. Can be null if span is in progress.
    """

    duration_ns = Column(
        BigInteger,
        Computed("end_time_unix_nano - start_time_unix_nano", persisted=True),
        nullable=True,
    )
    """
    Duration in nanoseconds: `BigInteger`. Computed from end_time - start_time.
    Stored as a persisted/stored generated column for efficient filtering.
    Will be NULL for in-progress spans (where end_time is NULL).
    """

    content = Column(Text, nullable=False)
    """
    Full span content as JSON: `Text`.
    Uses LONGTEXT in MySQL to support large spans (up to 4GB).
    """

    trace_info = relationship("SqlTraceInfo", backref=backref("spans", cascade="all"))
    """
    SQLAlchemy relationship (many:one) with :py:class:`mlflow.store.dbmodels.models.SqlTraceInfo`.
    """

    __table_args__ = (
        PrimaryKeyConstraint("trace_id", "span_id", name="spans_pk"),
        Index("index_spans_experiment_id", "experiment_id"),
        # Two indexes needed to support both filter patterns efficiently:
        Index(
            "index_spans_experiment_id_status_type", "experiment_id", "status", "type"
        ),  # For status-only and status+type filters
        Index(
            "index_spans_experiment_id_type_status", "experiment_id", "type", "status"
        ),  # For type-only and type+status filters
        Index("index_spans_experiment_id_duration", "experiment_id", "duration_ns"),
    )


class SqlEntityAssociation(Base):
    """
    DB model for entity associations.
    """

    __tablename__ = "entity_associations"

    association_id = Column(String(36), nullable=False)
    """
    Association ID: `String` (limit 36 characters).
    """

    source_type = Column(String(36), nullable=False)
    """
    Source entity type: `String` (limit 36 characters).
    """

    source_id = Column(String(36), nullable=False)
    """
    Source entity ID: `String` (limit 36 characters).
    """

    destination_type = Column(String(36), nullable=False)
    """
    Destination entity type: `String` (limit 36 characters).
    """

    destination_id = Column(String(36), nullable=False)
    """
    Destination entity ID: `String` (limit 36 characters).
    """

    created_time = Column(BigInteger, default=get_current_time_millis)
    """
    Creation time: `BigInteger`.
    """

    __table_args__ = (
        PrimaryKeyConstraint(
            "source_type",
            "source_id",
            "destination_type",
            "destination_id",
            name="entity_associations_pk",
        ),
        Index("index_entity_associations_association_id", "association_id"),
        Index(
            "index_entity_associations_reverse_lookup",
            "destination_type",
            "destination_id",
            "source_type",
            "source_id",
        ),
    )

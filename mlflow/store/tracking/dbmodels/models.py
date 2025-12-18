import json
import uuid
from typing import Any

import sqlalchemy as sa
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Computed,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    PrimaryKeyConstraint,
    String,
    Text,
    UnicodeText,
    UniqueConstraint,
)
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import backref, relationship

from mlflow.entities import (
    Assessment,
    AssessmentError,
    AssessmentSource,
    Dataset,
    DatasetRecord,
    DatasetRecordSource,
    EvaluationDataset,
    Expectation,
    Experiment,
    ExperimentTag,
    Feedback,
    GatewayEndpoint,
    GatewayEndpointBinding,
    GatewayEndpointModelMapping,
    GatewayEndpointTag,
    GatewayModelDefinition,
    GatewayResourceType,
    GatewaySecretInfo,
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
from mlflow.entities.dataset_record import DATASET_RECORD_WRAPPED_OUTPUT_KEY
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
from mlflow.utils.mlflow_tags import MLFLOW_USER, _get_run_name_from_tags
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


# Create MutableJSON type for tracking mutations in JSON columns
MutableJSON = MutableDict.as_mutable(JSON)


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
            if run_name := _get_run_name_from_tags(tags):
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


class SqlEvaluationDataset(Base):
    """
    DB model for evaluation datasets.
    """

    __tablename__ = "evaluation_datasets"

    dataset_id = Column(String(36), primary_key=True)
    """
    Dataset ID: `String` (limit 36 characters).
    *Primary Key* for ``evaluation_datasets`` table.
    """

    name = Column(String(255), nullable=False)
    """
    Dataset name: `String` (limit 255 characters). *Non null* in table schema.
    """

    schema = Column(Text, nullable=True)
    """
    Schema information: `Text`.
    """

    profile = Column(Text, nullable=True)
    """
    Profile information: `Text`.
    """

    digest = Column(String(64), nullable=True)
    """
    Dataset digest: `String` (limit 64 characters).
    """

    created_time = Column(BigInteger, default=get_current_time_millis)
    """
    Creation time: `BigInteger`.
    """

    last_update_time = Column(BigInteger, default=get_current_time_millis)
    """
    Last update time: `BigInteger`.
    """

    created_by = Column(String(255), nullable=True)
    """
    Creator user ID: `String` (limit 255 characters).
    """

    last_updated_by = Column(String(255), nullable=True)
    """
    Last updater user ID: `String` (limit 255 characters).
    """

    records = relationship(
        "SqlEvaluationDatasetRecord", back_populates="dataset", cascade="all, delete-orphan"
    )

    tags = relationship(
        "SqlEvaluationDatasetTag",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        PrimaryKeyConstraint("dataset_id", name="evaluation_datasets_pk"),
        Index("index_evaluation_datasets_name", "name"),
        Index("index_evaluation_datasets_created_time", "created_time"),
    )

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        Returns:
            :py:class:`mlflow.entities.EvaluationDataset`.
        """
        records = None
        # NB: Using SQLAlchemy's inspect module to determine if the field is loaded
        # or not as calling .records on the EvaluationDataset object will trigger
        # lazy-loading of the records.
        state = inspect(self)
        if "records" in state.dict:
            records = [record.to_mlflow_entity() for record in self.records]

        # Convert tags from relationship to dict
        # Since we use lazy="selectin", tags are always loaded
        # Return empty dict if no tags exist
        tags_dict = {tag.key: tag.value for tag in self.tags}

        dataset = EvaluationDataset(
            dataset_id=self.dataset_id,
            name=self.name,
            tags=tags_dict,
            schema=self.schema,
            profile=self.profile,
            digest=self.digest,
            created_time=self.created_time,
            last_update_time=self.last_update_time,
            created_by=self.created_by,
            last_updated_by=self.last_updated_by,
            # experiment_ids will be loaded lazily when accessed
        )

        if records is not None:
            dataset._records = records

        return dataset

    @classmethod
    def from_mlflow_entity(cls, dataset: EvaluationDataset):
        """
        Create SqlEvaluationDataset from EvaluationDataset entity.

        Args:
            dataset: EvaluationDataset entity

        Returns:
            SqlEvaluationDataset instance
        """
        # Note: tags are not set here - they are handled as
        # SqlEvaluationDatasetTag objects
        return cls(
            dataset_id=dataset.dataset_id,
            name=dataset.name,
            schema=dataset.schema,
            profile=dataset.profile,
            digest=dataset.digest,
            created_time=dataset.created_time or get_current_time_millis(),
            last_update_time=dataset.last_update_time or get_current_time_millis(),
            created_by=dataset.created_by,
            last_updated_by=dataset.last_updated_by,
        )


class SqlEvaluationDatasetTag(Base):
    """
    DB model for evaluation dataset tags.
    """

    __tablename__ = "evaluation_dataset_tags"

    dataset_id = Column(
        String(36),
        ForeignKey("evaluation_datasets.dataset_id", ondelete="CASCADE"),
        primary_key=True,
    )
    """
    Dataset ID: `String` (limit 36 characters). Foreign key to evaluation_datasets.
    *Primary Key* for ``evaluation_dataset_tags`` table.
    """

    key = Column(String(255), primary_key=True)
    """
    Tag key: `String` (limit 255 characters).
    *Primary Key* for ``evaluation_dataset_tags`` table.
    """

    value = Column(String(5000), nullable=True)
    """
    Tag value: `String` (limit 5000 characters).
    """

    __table_args__ = (
        PrimaryKeyConstraint("dataset_id", "key", name="evaluation_dataset_tags_pk"),
        ForeignKeyConstraint(
            ["dataset_id"],
            ["evaluation_datasets.dataset_id"],
            name="fk_evaluation_dataset_tags_dataset_id",
            ondelete="CASCADE",
        ),
        Index("index_evaluation_dataset_tags_dataset_id", "dataset_id"),
    )


class SqlEvaluationDatasetRecord(Base):
    """
    DB model for evaluation dataset records.
    """

    __tablename__ = "evaluation_dataset_records"
    RECORD_ID_PREFIX = "dr-"

    dataset_record_id = Column(String(36), primary_key=True)
    """
    Dataset record ID: `String` (limit 36 characters).
    *Primary Key* for ``evaluation_dataset_records`` table.
    """

    dataset_id = Column(
        String(36), ForeignKey("evaluation_datasets.dataset_id", ondelete="CASCADE"), nullable=False
    )
    """
    Dataset ID: `String` (limit 36 characters). Foreign key to evaluation_datasets.
    """

    inputs = Column(MutableJSON, nullable=False)
    """
    Inputs JSON: `JSON`. *Non null* in table schema.
    """

    outputs = Column(MutableJSON, nullable=True)
    """
    Outputs JSON: `JSON`.
    """

    expectations = Column(MutableJSON, nullable=True)
    """
    Expectations JSON: `JSON`.
    """

    tags = Column(MutableJSON, nullable=True)
    """
    Tags JSON: `JSON`.
    """

    source = Column(MutableJSON, nullable=True)
    """
    Source JSON: `JSON`.
    """

    source_id = Column(String(36), nullable=True)
    """
    Source ID for lookups: `String` (limit 36 characters).
    """

    source_type = Column(String(255), nullable=True)
    """
    Source type: `Text`.
    """

    created_time = Column(BigInteger, default=get_current_time_millis)
    """
    Creation time: `BigInteger`.
    """

    last_update_time = Column(BigInteger, default=get_current_time_millis)
    """
    Last update time: `BigInteger`.
    """

    created_by = Column(String(255), nullable=True)
    """
    Creator user ID: `String` (limit 255 characters).
    """

    last_updated_by = Column(String(255), nullable=True)
    """
    Last updater user ID: `String` (limit 255 characters).
    """

    input_hash = Column(String(64), nullable=False)
    """
    Hash of inputs for deduplication: `String` (limit 64 characters).
    """

    dataset = relationship("SqlEvaluationDataset", back_populates="records")

    __table_args__ = (
        PrimaryKeyConstraint("dataset_record_id", name="evaluation_dataset_records_pk"),
        Index("index_evaluation_dataset_records_dataset_id", "dataset_id"),
        UniqueConstraint("dataset_id", "input_hash", name="unique_dataset_input"),
        ForeignKeyConstraint(
            ["dataset_id"],
            ["evaluation_datasets.dataset_id"],
            name="fk_evaluation_dataset_records_dataset_id",
            ondelete="CASCADE",
        ),
    )

    def __init__(self, **kwargs):
        """Initialize a new dataset record with auto-generated ID if not provided."""
        if "dataset_record_id" not in kwargs:
            kwargs["dataset_record_id"] = self.generate_record_id()
        super().__init__(**kwargs)

    @staticmethod
    def generate_record_id() -> str:
        """
        Generate a unique ID for dataset records.

        Returns:
            A unique record ID with the format "dr-<uuid_hex>".
        """
        return f"{SqlEvaluationDatasetRecord.RECORD_ID_PREFIX}{uuid.uuid4().hex}"

    def to_mlflow_entity(self):
        inputs = self.inputs
        expectations = self.expectations
        tags = self.tags

        outputs = self.outputs.get(DATASET_RECORD_WRAPPED_OUTPUT_KEY) if self.outputs else None

        source = None
        if self.source:
            source = DatasetRecordSource.from_dict(self.source)

        return DatasetRecord(
            dataset_record_id=self.dataset_record_id,
            dataset_id=self.dataset_id,
            inputs=inputs,
            outputs=outputs,
            expectations=expectations,
            tags=tags,
            source=source,
            source_id=self.source_id,
            created_time=self.created_time,
            last_update_time=self.last_update_time,
            created_by=self.created_by,
            last_updated_by=self.last_updated_by,
        )

    @classmethod
    def from_mlflow_entity(cls, record: DatasetRecord, input_hash: str):
        """
        Create SqlEvaluationDatasetRecord from DatasetRecord entity.

        Args:
            record: DatasetRecord entity
            input_hash: SHA256 hash of inputs for deduplication

        Returns:
            SqlEvaluationDatasetRecord instance
        """

        source_dict = None
        if record.source:
            source_dict = record.source.to_dict()

        outputs = (
            {DATASET_RECORD_WRAPPED_OUTPUT_KEY: record.outputs}
            if record.outputs is not None
            else None
        )

        kwargs = {
            "dataset_id": record.dataset_id,
            "inputs": record.inputs,
            "outputs": outputs,
            "expectations": record.expectations,
            "tags": record.tags,
            "source": source_dict,
            "source_id": record.source_id,
            "source_type": record.source.source_type if record.source else None,
            "created_time": record.created_time or get_current_time_millis(),
            "last_update_time": record.last_update_time or get_current_time_millis(),
            "created_by": record.created_by,
            "last_updated_by": record.last_updated_by,
            "input_hash": input_hash,
        }

        if record.dataset_record_id:
            kwargs["dataset_record_id"] = record.dataset_record_id

        return cls(**kwargs)

    def merge(self, new_record_dict: dict[str, Any]) -> None:
        """
        Merge new record data into this existing record.

        Updates outputs, expectations and tags by merging new values with existing ones.
        Preserves created_time and created_by from the original record.

        Args:
            new_record_dict: Dictionary containing new record data with optional
                           'outputs', 'expectations' and 'tags' fields to merge.
        """
        if "outputs" in new_record_dict:
            new_outputs = new_record_dict["outputs"]
            self.outputs = (
                {DATASET_RECORD_WRAPPED_OUTPUT_KEY: new_outputs}
                if new_outputs is not None
                else None
            )

        if new_expectations := new_record_dict.get("expectations"):
            if self.expectations is None:
                self.expectations = {}
            self.expectations.update(new_expectations)

        if new_tags := new_record_dict.get("tags"):
            if self.tags is None:
                self.tags = {}
            self.tags.update(new_tags)

        self.last_update_time = get_current_time_millis()

        # Update last_updated_by if mlflow.user tag is present
        # Otherwise keep the existing last_updated_by (don't change it to None)
        if new_tags and MLFLOW_USER in new_tags:
            self.last_updated_by = new_tags[MLFLOW_USER]


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
    ASSOCIATION_ID_PREFIX = "a-"

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

    def __init__(self, **kwargs):
        """Initialize a new entity association with auto-generated ID if not provided."""
        if "association_id" not in kwargs:
            kwargs["association_id"] = self.generate_association_id()
        super().__init__(**kwargs)

    @staticmethod
    def generate_association_id() -> str:
        """
        Generate a unique ID for entity associations.

        Returns:
            A unique association ID with the format "a-<uuid_hex>".
        """
        return f"{SqlEntityAssociation.ASSOCIATION_ID_PREFIX}{uuid.uuid4().hex}"


class SqlScorer(Base):
    """
    DB model for storing scorer information. These are recorded in ``scorers`` table.
    """

    __tablename__ = "scorers"

    experiment_id = Column(
        Integer, ForeignKey("experiments.experiment_id", ondelete="CASCADE"), nullable=False
    )
    """
    Experiment ID to which this scorer belongs: *Foreign Key* into ``experiments`` table.
    """
    scorer_name = Column(String(256), nullable=False)
    """
    Scorer name: `String` (limit 256 characters). Part of *Primary Key* for ``scorers`` table.
    """
    scorer_id = Column(String(36), nullable=False)
    """
    Scorer ID: `String` (limit 36 characters). Unique identifier for the scorer.
    """

    experiment = relationship("SqlExperiment", backref=backref("scorers", cascade="all"))
    """
    SQLAlchemy relationship (many:one) with :py:class:`mlflow.store.dbmodels.models.SqlExperiment`.
    """

    __table_args__ = (
        PrimaryKeyConstraint("scorer_id", name="scorer_pk"),
        Index(
            f"index_{__tablename__}_experiment_id_scorer_name",
            "experiment_id",
            "scorer_name",
            unique=True,
        ),
    )

    def __repr__(self):
        return f"<SqlScorer ({self.experiment_id}, {self.scorer_name}, {self.scorer_id})>"


class SqlScorerVersion(Base):
    """
    DB model for storing scorer version information. These are recorded in
    ``scorer_versions`` table.
    """

    __tablename__ = "scorer_versions"

    scorer_id = Column(
        String(36), ForeignKey("scorers.scorer_id", ondelete="CASCADE"), nullable=False
    )
    """
    Scorer ID: `String` (limit 36 characters). *Foreign Key* into ``scorers`` table.
    """
    scorer_version = Column(Integer, nullable=False)
    """
    Scorer version: `Integer`. Part of *Primary Key* for ``scorer_versions`` table.
    """
    serialized_scorer = Column(Text, nullable=False)
    """
    Serialized scorer data: `Text`. Contains the serialized scorer object.
    """
    creation_time = Column(BigInteger(), default=get_current_time_millis)
    """
    Creation time of scorer version: `BigInteger`. Automatically set to current time when created.
    """

    # Relationship to the parent scorer
    scorer = relationship("SqlScorer", backref=backref("scorer_versions", cascade="all"))
    """
    SQLAlchemy relationship (many:one) with :py:class:`mlflow.store.dbmodels.models.SqlScorer`.
    """

    __table_args__ = (
        PrimaryKeyConstraint("scorer_id", "scorer_version", name="scorer_version_pk"),
        Index(f"index_{__tablename__}_scorer_id", "scorer_id"),
    )

    def __repr__(self):
        return f"<SqlScorerVersion ({self.scorer_id}, {self.scorer_version})>"

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        Returns:
            mlflow.entities.ScorerVersion.
        """
        from mlflow.entities.scorer import ScorerVersion

        return ScorerVersion(
            experiment_id=str(self.scorer.experiment_id),
            scorer_name=self.scorer.scorer_name,
            scorer_version=self.scorer_version,
            serialized_scorer=self.serialized_scorer,
            creation_time=self.creation_time,
            scorer_id=self.scorer_id,
        )


class SqlJob(Base):
    """
    DB model for Job entities. These are recorded in the ``jobs`` table.
    """

    __tablename__ = "jobs"

    id = Column(String(36), nullable=False)
    """
    Job ID: `String` (limit 36 characters). *Primary Key* for ``jobs`` table.
    """

    creation_time = Column(BigInteger, default=get_current_time_millis, nullable=False)
    """
    Creation timestamp: `BigInteger`.
    """

    function_fullname = Column(String(500), nullable=False)
    """
    Function fullname: `String` (limit 500 characters).
    """

    params = Column(Text, nullable=False)
    """
    Job parameters: `Text`.
    """

    timeout = Column(sa.types.Float(precision=53), nullable=True)
    """
    Job execution timeout in seconds: `Float`
    """

    status = Column(Integer, nullable=False)
    """
    Job status: `Integer`.
    """

    result = Column(Text, nullable=True)
    """
    Job result: `Text`.
    """

    retry_count = Column(Integer, default=0, nullable=False)
    """
    Job retry count: `Integer`
    """

    last_update_time = Column(BigInteger(), default=get_current_time_millis, nullable=False)
    """
    Last Update time of experiment: `BigInteger`.
    """

    __table_args__ = (
        PrimaryKeyConstraint("id", name="jobs_pk"),
        Index(
            "index_jobs_function_status_creation_time",
            "function_fullname",
            "status",
            "creation_time",
        ),
    )

    def __repr__(self):
        return f"<SqlJob ({self.id}, {self.function_fullname}, {self.status})>"

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        Returns:
            mlflow.entities._job.Job.
        """
        from mlflow.entities._job import Job
        from mlflow.entities._job_status import JobStatus

        return Job(
            job_id=self.id,
            creation_time=self.creation_time,
            function_fullname=self.function_fullname,
            params=self.params,
            timeout=self.timeout,
            status=JobStatus.from_int(self.status),
            result=self.result,
            retry_count=self.retry_count,
            last_update_time=self.last_update_time,
        )


class SqlGatewaySecret(Base):
    """
    DB model for secrets. These are recorded in the ``secrets`` table.
    Stores encrypted credentials used by MLflow resources (e.g., LLM provider API keys).
    """

    __tablename__ = "secrets"

    secret_id = Column(String(36), nullable=False)
    """
    Secret ID: `String` (limit 36 characters). *Primary Key* for ``secrets`` table.

    NB: IMMUTABLE. This field is used as part of the AAD (Additional Authenticated Data) during
    AES-GCM encryption. If modified, decryption will fail with authentication error. See
    mlflow/utils/crypto.py:_create_aad() for details.
    """
    secret_name = Column(String(255), nullable=False)
    """
    Secret name: `String` (limit 255 characters). User-provided name for the secret.
    Defined as *Unique* in table schema to prevent confusing selection of secrets in the UI.

    NB: IMMUTABLE. This field is used as part of the AAD (Additional Authenticated Data) during
    AES-GCM encryption. If modified, decryption will fail with authentication error. To "rename"
    a secret, create a new secret with the desired name and delete the old one. See
    mlflow/utils/crypto.py:_create_aad() for details.
    """
    encrypted_value = Column(LargeBinary, nullable=False)
    """
    Encrypted secret data: `LargeBinary`. Combined nonce (12 bytes) + AES-GCM ciphertext +
    tag (16 bytes). The secret value is encrypted using envelope encryption with a DEK, and
    the nonce is prepended for storage. AAD (Additional Authenticated Data) from secret_id
    and secret_name is included during encryption to prevent ciphertext substitution attacks.
    """
    wrapped_dek = Column(LargeBinary, nullable=False)
    """
    Wrapped data encryption key: `LargeBinary`. DEK encrypted by KEK.
    The DEK is a randomly generated 256-bit AES key used to encrypt the secret value.
    """
    kek_version = Column(Integer, nullable=False, default=1)
    """
    KEK version: `Integer`. Indicates which KEK version was used to wrap the DEK.
    Used for KEK rotation - allows multiple KEK versions to coexist during migration.
    """
    masked_value = Column(String(500), nullable=False)
    """
    Masked secret value: `String` (limit 500 characters). JSON-serialized dict showing partial
    secret values for identification. Format: ``{"key": "prefix...suffix"}``, e.g.,
    ``{"api_key": "sk-...xyz123"}`` or ``{"aws_access_key_id": "AKI...1234", ...}``.
    Helps users identify secrets without exposing the full values.
    """
    provider = Column(String(64), nullable=True)
    """
    Provider identifier: `String` (limit 64 characters). Optional.
    E.g., "anthropic", "openai", "cohere", "vertex_ai", "bedrock", "databricks".
    """
    auth_config = Column(Text, nullable=True)
    """
    Provider authentication config: `Text` (JSON string). Non-sensitive metadata for
    provider configuration like region, project_id, endpoint URL. Useful for UI display
    and disambiguation. Not encrypted since it contains no secrets.
    For multi-auth providers, includes "auth_mode" key (e.g., "access_keys", "iam_role").
    """
    description = Column(Text, nullable=True)
    """
    Secret description: `Text`. Optional user-provided description for the API key.
    """
    created_by = Column(String(255), nullable=True)
    """
    Creator user ID: `String` (limit 255 characters).
    """
    created_at = Column(BigInteger, default=get_current_time_millis, nullable=False)
    """
    Creation timestamp: `BigInteger`.
    """
    last_updated_by = Column(String(255), nullable=True)
    """
    Last updater user ID: `String` (limit 255 characters).
    """
    last_updated_at = Column(BigInteger, default=get_current_time_millis, nullable=False)
    """
    Last update timestamp: `BigInteger`.
    """

    __table_args__ = (
        PrimaryKeyConstraint("secret_id", name="secrets_pk"),
        Index("unique_secret_name", "secret_name", unique=True),
    )

    def __repr__(self):
        return f"<SqlGatewaySecret ({self.secret_id}, {self.secret_name})>"

    def to_mlflow_entity(self):
        try:
            masked_value = json.loads(self.masked_value)
        except (json.JSONDecodeError, TypeError):
            masked_value = {"value": "***"}

        return GatewaySecretInfo(
            secret_id=self.secret_id,
            secret_name=self.secret_name,
            masked_values=masked_value,
            created_at=self.created_at,
            last_updated_at=self.last_updated_at,
            provider=self.provider,
            auth_config=json.loads(self.auth_config) if self.auth_config else None,
            created_by=self.created_by,
            last_updated_by=self.last_updated_by,
        )


class SqlGatewayEndpoint(Base):
    """
    DB model for endpoints. These are recorded in ``endpoints`` table.
    Represents LLM gateway endpoints that route requests to configured models.
    """

    __tablename__ = "endpoints"

    endpoint_id = Column(String(36), nullable=False)
    """
    Endpoint ID: `String` (limit 36 characters). *Primary Key* for ``endpoints`` table.
    """
    name = Column(String(255), nullable=True)
    """
    Endpoint name: `String` (limit 255 characters). User-provided name for the endpoint.
    Defined as *Unique* in table schema.
    """
    created_by = Column(String(255), nullable=True)
    """
    Creator user ID: `String` (limit 255 characters).
    """
    created_at = Column(BigInteger, default=get_current_time_millis, nullable=False)
    """
    Creation timestamp: `BigInteger`.
    """
    last_updated_by = Column(String(255), nullable=True)
    """
    Last updater user ID: `String` (limit 255 characters).
    """
    last_updated_at = Column(BigInteger, default=get_current_time_millis, nullable=False)
    """
    Last update timestamp: `BigInteger`.
    """

    __table_args__ = (
        PrimaryKeyConstraint("endpoint_id", name="endpoints_pk"),
        Index("unique_endpoint_name", "name", unique=True),
    )

    def __repr__(self):
        return f"<SqlGatewayEndpoint ({self.endpoint_id}, {self.name})>"

    def to_mlflow_entity(self):
        return GatewayEndpoint(
            endpoint_id=self.endpoint_id,
            name=self.name,
            model_mappings=[mapping.to_mlflow_entity() for mapping in self.model_mappings],
            tags=[tag.to_mlflow_entity() for tag in self.tags],
            created_at=self.created_at,
            last_updated_at=self.last_updated_at,
            created_by=self.created_by,
            last_updated_by=self.last_updated_by,
        )


class SqlGatewayModelDefinition(Base):
    """
    DB model for model definitions. These are recorded in ``model_definitions`` table.
    Represents reusable LLM model configurations that can be shared across multiple endpoints.
    """

    __tablename__ = "model_definitions"

    model_definition_id = Column(String(36), nullable=False)
    """
    Model Definition ID: `String` (limit 36 characters).
    *Primary Key* for ``model_definitions`` table.
    """
    name = Column(String(255), nullable=False)
    """
    Model definition name: `String` (limit 255 characters). User-provided name for identification.
    Defined as *Unique* in table schema.
    """
    secret_id = Column(
        String(36), ForeignKey("secrets.secret_id", ondelete="SET NULL"), nullable=True
    )
    """
    Secret ID: `String` (limit 36 characters). *Foreign Key* into ``secrets`` table.
    References the API key/credentials for this model. Nullable to allow orphaned
    model definitions when secrets are deleted.
    """
    provider = Column(String(64), nullable=False)
    """
    Provider identifier: `String` (limit 64 characters).
    E.g., "anthropic", "openai", "cohere", "vertex_ai", "bedrock", "databricks".
    """
    model_name = Column(String(256), nullable=False)
    """
    Model name: `String` (limit 256 characters). Provider-specific model identifier.
    E.g., "claude-3-5-sonnet-20241022", "gpt-4o", "command-r-plus".
    """
    created_by = Column(String(255), nullable=True)
    """
    Creator user ID: `String` (limit 255 characters).
    """
    created_at = Column(BigInteger, default=get_current_time_millis, nullable=False)
    """
    Creation timestamp: `BigInteger`.
    """
    last_updated_by = Column(String(255), nullable=True)
    """
    Last updater user ID: `String` (limit 255 characters).
    """
    last_updated_at = Column(BigInteger, default=get_current_time_millis, nullable=False)
    """
    Last update timestamp: `BigInteger`.
    """

    secret = relationship("SqlGatewaySecret")
    """
    SQLAlchemy relationship (many:one) with
    :py:class:`mlflow.store.tracking.dbmodels.models.SqlGatewaySecret`.
    """

    __table_args__ = (
        PrimaryKeyConstraint("model_definition_id", name="model_definitions_pk"),
        Index("unique_model_definition_name", "name", unique=True),
        Index("index_model_definitions_secret_id", "secret_id"),
        Index("index_model_definitions_provider", "provider"),
    )

    def __repr__(self):
        return f"<SqlGatewayModelDefinition ({self.model_definition_id}, {self.name})>"

    def to_mlflow_entity(self):
        return GatewayModelDefinition(
            model_definition_id=self.model_definition_id,
            name=self.name,
            secret_id=self.secret_id,
            secret_name=self.secret.secret_name if self.secret else None,
            provider=self.provider,
            model_name=self.model_name,
            created_at=self.created_at,
            last_updated_at=self.last_updated_at,
            created_by=self.created_by,
            last_updated_by=self.last_updated_by,
        )


class SqlGatewayEndpointModelMapping(Base):
    """
    DB model for endpoint-model mappings. These are recorded in ``endpoint_model_mappings`` table.
    Junction table linking endpoints to model definitions (supports multi-model routing).
    """

    __tablename__ = "endpoint_model_mappings"

    mapping_id = Column(String(36), nullable=False)
    """
    Mapping ID: `String` (limit 36 characters). *Primary Key* for ``endpoint_model_mappings`` table.
    """
    endpoint_id = Column(
        String(36), ForeignKey("endpoints.endpoint_id", ondelete="CASCADE"), nullable=False
    )
    """
    Endpoint ID: `String` (limit 36 characters). *Foreign Key* into ``endpoints`` table.
    Cascades on delete - removing an endpoint removes all its model mappings.
    """
    model_definition_id = Column(
        String(36),
        ForeignKey("model_definitions.model_definition_id"),
        nullable=False,
    )
    """
    Model Definition ID: `String` (limit 36 characters).
    *Foreign Key* into ``model_definitions`` table.
    Prevents deletion of a model definition that is in use (default FK behavior).
    """
    weight = Column(Float, default=1.0, nullable=False)
    """
    Routing weight: `Float`. Used for traffic distribution when endpoint has multiple models.
    Default is 1.0.
    """
    created_by = Column(String(255), nullable=True)
    """
    Creator user ID: `String` (limit 255 characters).
    """
    created_at = Column(BigInteger, default=get_current_time_millis, nullable=False)
    """
    Creation timestamp: `BigInteger`.
    """

    endpoint = relationship("SqlGatewayEndpoint", backref=backref("model_mappings", cascade="all"))
    """
    SQLAlchemy relationship (many:one) with
    :py:class:`mlflow.store.tracking.dbmodels.models.SqlGatewayEndpoint`.
    """
    model_definition = relationship("SqlGatewayModelDefinition")
    """
    SQLAlchemy relationship (many:one) with
    :py:class:`mlflow.store.tracking.dbmodels.models.SqlGatewayModelDefinition`.
    """

    __table_args__ = (
        PrimaryKeyConstraint("mapping_id", name="endpoint_model_mappings_pk"),
        Index("index_endpoint_model_mappings_endpoint_id", "endpoint_id"),
        Index("index_endpoint_model_mappings_model_definition_id", "model_definition_id"),
        Index(
            "unique_endpoint_model_mapping",
            "endpoint_id",
            "model_definition_id",
            unique=True,
        ),
    )

    def __repr__(self):
        return (
            f"<SqlGatewayEndpointModelMapping ({self.mapping_id}, "
            f"endpoint={self.endpoint_id}, model={self.model_definition_id})>"
        )

    def to_mlflow_entity(self):
        model_def = None
        if self.model_definition:
            model_def = self.model_definition.to_mlflow_entity()
        return GatewayEndpointModelMapping(
            mapping_id=self.mapping_id,
            endpoint_id=self.endpoint_id,
            model_definition_id=self.model_definition_id,
            model_definition=model_def,
            weight=self.weight,
            created_at=self.created_at,
            created_by=self.created_by,
        )


class SqlGatewayEndpointBinding(Base):
    """
    DB model for endpoint bindings. These are recorded in ``endpoint_bindings`` table.
    Tracks which resources are bound to which endpoints (e.g., model configurations, experiments).
    """

    __tablename__ = "endpoint_bindings"

    endpoint_id = Column(
        String(36), ForeignKey("endpoints.endpoint_id", ondelete="CASCADE"), nullable=False
    )
    """
    Endpoint ID: `String` (limit 36 characters). *Foreign Key* into ``endpoints`` table.
    Cascades on delete. Part of composite primary key.
    """
    resource_type = Column(String(50), nullable=False)
    """
    Resource type: `String` (limit 50 characters). Type of resource bound to the endpoint.
    E.g., "endpoint_model", "experiment", "registered_model". Part of composite primary key.
    """
    resource_id = Column(String(255), nullable=False)
    """
    Resource ID: `String` (limit 255 characters). ID of the specific resource instance.
    Part of composite primary key.
    """
    created_at = Column(BigInteger, default=get_current_time_millis, nullable=False)
    """
    Creation timestamp: `BigInteger`.
    """
    created_by = Column(String(255), nullable=True)
    """
    Creator user ID: `String` (limit 255 characters).
    """
    last_updated_at = Column(BigInteger, default=get_current_time_millis, nullable=False)
    """
    Last update timestamp: `BigInteger`.
    """
    last_updated_by = Column(String(255), nullable=True)
    """
    Last updater user ID: `String` (limit 255 characters).
    """

    endpoint = relationship("SqlGatewayEndpoint", backref=backref("bindings", cascade="all"))
    """
    SQLAlchemy relationship (many:one) with
    :py:class:`mlflow.store.tracking.dbmodels.models.SqlGatewayEndpoint`.
    """

    __table_args__ = (
        PrimaryKeyConstraint(
            "endpoint_id", "resource_type", "resource_id", name="endpoint_bindings_pk"
        ),
    )

    def __repr__(self):
        return (
            f"<SqlGatewayEndpointBinding "
            f"({self.endpoint_id}, {self.resource_type}, {self.resource_id})>"
        )

    def to_mlflow_entity(self):
        return GatewayEndpointBinding(
            endpoint_id=self.endpoint_id,
            resource_type=GatewayResourceType(self.resource_type),
            resource_id=self.resource_id,
            created_at=self.created_at,
            last_updated_at=self.last_updated_at,
            created_by=self.created_by,
            last_updated_by=self.last_updated_by,
        )


class SqlGatewayEndpointTag(Base):
    """
    DB model for endpoint tags. These are recorded in ``endpoint_tags`` table.
    Tags are key-value pairs associated with endpoints for categorization and filtering.
    """

    __tablename__ = "endpoint_tags"

    key = Column(String(250), nullable=False)
    """
    Tag key: `String` (limit 250 characters). Part of composite *Primary Key*.
    """
    value = Column(String(5000), nullable=True)
    """
    Value associated with tag: `String` (limit 5000 characters). Could be *null*.
    """
    endpoint_id = Column(
        String(36), ForeignKey("endpoints.endpoint_id", ondelete="CASCADE"), nullable=False
    )
    """
    Endpoint ID to which this tag belongs: *Foreign Key* into ``endpoints`` table.
    Part of composite *Primary Key*. Cascades on delete.
    """
    endpoint = relationship("SqlGatewayEndpoint", backref=backref("tags", cascade="all"))
    """
    SQLAlchemy relationship (many:one) with
    :py:class:`mlflow.store.tracking.dbmodels.models.SqlGatewayEndpoint`.
    """

    __table_args__ = (
        PrimaryKeyConstraint("key", "endpoint_id", name="endpoint_tag_pk"),
        Index("index_endpoint_tags_endpoint_id", "endpoint_id"),
    )

    def __repr__(self):
        return f"<SqlGatewayEndpointTag({self.key}, {self.value})>"

    def to_mlflow_entity(self):
        return GatewayEndpointTag(key=self.key, value=self.value)

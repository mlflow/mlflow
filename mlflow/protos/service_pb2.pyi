from scalapb import scalapb_pb2 as _scalapb_pb2
import databricks_pb2 as _databricks_pb2
from google.protobuf import duration_pb2 as _duration_pb2
from google.protobuf import field_mask_pb2 as _field_mask_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
import assessments_pb2 as _assessments_pb2
import datasets_pb2 as _datasets_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import service as _service
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ViewType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTIVE_ONLY: _ClassVar[ViewType]
    DELETED_ONLY: _ClassVar[ViewType]
    ALL: _ClassVar[ViewType]

class SourceType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NOTEBOOK: _ClassVar[SourceType]
    JOB: _ClassVar[SourceType]
    PROJECT: _ClassVar[SourceType]
    LOCAL: _ClassVar[SourceType]
    UNKNOWN: _ClassVar[SourceType]

class RunStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RUNNING: _ClassVar[RunStatus]
    SCHEDULED: _ClassVar[RunStatus]
    FINISHED: _ClassVar[RunStatus]
    FAILED: _ClassVar[RunStatus]
    KILLED: _ClassVar[RunStatus]

class TraceStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRACE_STATUS_UNSPECIFIED: _ClassVar[TraceStatus]
    OK: _ClassVar[TraceStatus]
    ERROR: _ClassVar[TraceStatus]
    IN_PROGRESS: _ClassVar[TraceStatus]

class LoggedModelStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LOGGED_MODEL_STATUS_UNSPECIFIED: _ClassVar[LoggedModelStatus]
    LOGGED_MODEL_PENDING: _ClassVar[LoggedModelStatus]
    LOGGED_MODEL_READY: _ClassVar[LoggedModelStatus]
    LOGGED_MODEL_UPLOAD_FAILED: _ClassVar[LoggedModelStatus]
ACTIVE_ONLY: ViewType
DELETED_ONLY: ViewType
ALL: ViewType
NOTEBOOK: SourceType
JOB: SourceType
PROJECT: SourceType
LOCAL: SourceType
UNKNOWN: SourceType
RUNNING: RunStatus
SCHEDULED: RunStatus
FINISHED: RunStatus
FAILED: RunStatus
KILLED: RunStatus
TRACE_STATUS_UNSPECIFIED: TraceStatus
OK: TraceStatus
ERROR: TraceStatus
IN_PROGRESS: TraceStatus
LOGGED_MODEL_STATUS_UNSPECIFIED: LoggedModelStatus
LOGGED_MODEL_PENDING: LoggedModelStatus
LOGGED_MODEL_READY: LoggedModelStatus
LOGGED_MODEL_UPLOAD_FAILED: LoggedModelStatus

class Metric(_message.Message):
    __slots__ = ("key", "value", "timestamp", "step", "dataset_name", "dataset_digest", "model_id", "run_id")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    DATASET_NAME_FIELD_NUMBER: _ClassVar[int]
    DATASET_DIGEST_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: float
    timestamp: int
    step: int
    dataset_name: str
    dataset_digest: str
    model_id: str
    run_id: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ..., timestamp: _Optional[int] = ..., step: _Optional[int] = ..., dataset_name: _Optional[str] = ..., dataset_digest: _Optional[str] = ..., model_id: _Optional[str] = ..., run_id: _Optional[str] = ...) -> None: ...

class Param(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class Run(_message.Message):
    __slots__ = ("info", "data", "inputs", "outputs")
    INFO_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    info: RunInfo
    data: RunData
    inputs: RunInputs
    outputs: RunOutputs
    def __init__(self, info: _Optional[_Union[RunInfo, _Mapping]] = ..., data: _Optional[_Union[RunData, _Mapping]] = ..., inputs: _Optional[_Union[RunInputs, _Mapping]] = ..., outputs: _Optional[_Union[RunOutputs, _Mapping]] = ...) -> None: ...

class RunData(_message.Message):
    __slots__ = ("metrics", "params", "tags")
    METRICS_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    metrics: _containers.RepeatedCompositeFieldContainer[Metric]
    params: _containers.RepeatedCompositeFieldContainer[Param]
    tags: _containers.RepeatedCompositeFieldContainer[RunTag]
    def __init__(self, metrics: _Optional[_Iterable[_Union[Metric, _Mapping]]] = ..., params: _Optional[_Iterable[_Union[Param, _Mapping]]] = ..., tags: _Optional[_Iterable[_Union[RunTag, _Mapping]]] = ...) -> None: ...

class RunInputs(_message.Message):
    __slots__ = ("dataset_inputs", "model_inputs")
    DATASET_INPUTS_FIELD_NUMBER: _ClassVar[int]
    MODEL_INPUTS_FIELD_NUMBER: _ClassVar[int]
    dataset_inputs: _containers.RepeatedCompositeFieldContainer[DatasetInput]
    model_inputs: _containers.RepeatedCompositeFieldContainer[ModelInput]
    def __init__(self, dataset_inputs: _Optional[_Iterable[_Union[DatasetInput, _Mapping]]] = ..., model_inputs: _Optional[_Iterable[_Union[ModelInput, _Mapping]]] = ...) -> None: ...

class RunOutputs(_message.Message):
    __slots__ = ("model_outputs",)
    MODEL_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    model_outputs: _containers.RepeatedCompositeFieldContainer[ModelOutput]
    def __init__(self, model_outputs: _Optional[_Iterable[_Union[ModelOutput, _Mapping]]] = ...) -> None: ...

class RunTag(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class ExperimentTag(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class RunInfo(_message.Message):
    __slots__ = ("run_id", "run_uuid", "run_name", "experiment_id", "user_id", "status", "start_time", "end_time", "artifact_uri", "lifecycle_stage")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_UUID_FIELD_NUMBER: _ClassVar[int]
    RUN_NAME_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_URI_FIELD_NUMBER: _ClassVar[int]
    LIFECYCLE_STAGE_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    run_uuid: str
    run_name: str
    experiment_id: str
    user_id: str
    status: RunStatus
    start_time: int
    end_time: int
    artifact_uri: str
    lifecycle_stage: str
    def __init__(self, run_id: _Optional[str] = ..., run_uuid: _Optional[str] = ..., run_name: _Optional[str] = ..., experiment_id: _Optional[str] = ..., user_id: _Optional[str] = ..., status: _Optional[_Union[RunStatus, str]] = ..., start_time: _Optional[int] = ..., end_time: _Optional[int] = ..., artifact_uri: _Optional[str] = ..., lifecycle_stage: _Optional[str] = ...) -> None: ...

class Experiment(_message.Message):
    __slots__ = ("experiment_id", "name", "artifact_location", "lifecycle_stage", "last_update_time", "creation_time", "tags")
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_LOCATION_FIELD_NUMBER: _ClassVar[int]
    LIFECYCLE_STAGE_FIELD_NUMBER: _ClassVar[int]
    LAST_UPDATE_TIME_FIELD_NUMBER: _ClassVar[int]
    CREATION_TIME_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    name: str
    artifact_location: str
    lifecycle_stage: str
    last_update_time: int
    creation_time: int
    tags: _containers.RepeatedCompositeFieldContainer[ExperimentTag]
    def __init__(self, experiment_id: _Optional[str] = ..., name: _Optional[str] = ..., artifact_location: _Optional[str] = ..., lifecycle_stage: _Optional[str] = ..., last_update_time: _Optional[int] = ..., creation_time: _Optional[int] = ..., tags: _Optional[_Iterable[_Union[ExperimentTag, _Mapping]]] = ...) -> None: ...

class DatasetInput(_message.Message):
    __slots__ = ("tags", "dataset")
    TAGS_FIELD_NUMBER: _ClassVar[int]
    DATASET_FIELD_NUMBER: _ClassVar[int]
    tags: _containers.RepeatedCompositeFieldContainer[InputTag]
    dataset: Dataset
    def __init__(self, tags: _Optional[_Iterable[_Union[InputTag, _Mapping]]] = ..., dataset: _Optional[_Union[Dataset, _Mapping]] = ...) -> None: ...

class ModelInput(_message.Message):
    __slots__ = ("model_id",)
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    def __init__(self, model_id: _Optional[str] = ...) -> None: ...

class InputTag(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class Dataset(_message.Message):
    __slots__ = ("name", "digest", "source_type", "source", "schema", "profile")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    SOURCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    PROFILE_FIELD_NUMBER: _ClassVar[int]
    name: str
    digest: str
    source_type: str
    source: str
    schema: str
    profile: str
    def __init__(self, name: _Optional[str] = ..., digest: _Optional[str] = ..., source_type: _Optional[str] = ..., source: _Optional[str] = ..., schema: _Optional[str] = ..., profile: _Optional[str] = ...) -> None: ...

class ModelOutput(_message.Message):
    __slots__ = ("model_id", "step")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    step: int
    def __init__(self, model_id: _Optional[str] = ..., step: _Optional[int] = ...) -> None: ...

class CreateExperiment(_message.Message):
    __slots__ = ("name", "artifact_location", "tags")
    class Response(_message.Message):
        __slots__ = ("experiment_id",)
        EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
        experiment_id: str
        def __init__(self, experiment_id: _Optional[str] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_LOCATION_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    name: str
    artifact_location: str
    tags: _containers.RepeatedCompositeFieldContainer[ExperimentTag]
    def __init__(self, name: _Optional[str] = ..., artifact_location: _Optional[str] = ..., tags: _Optional[_Iterable[_Union[ExperimentTag, _Mapping]]] = ...) -> None: ...

class SearchExperiments(_message.Message):
    __slots__ = ("max_results", "page_token", "filter", "order_by", "view_type")
    class Response(_message.Message):
        __slots__ = ("experiments", "next_page_token")
        EXPERIMENTS_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        experiments: _containers.RepeatedCompositeFieldContainer[Experiment]
        next_page_token: str
        def __init__(self, experiments: _Optional[_Iterable[_Union[Experiment, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    ORDER_BY_FIELD_NUMBER: _ClassVar[int]
    VIEW_TYPE_FIELD_NUMBER: _ClassVar[int]
    max_results: int
    page_token: str
    filter: str
    order_by: _containers.RepeatedScalarFieldContainer[str]
    view_type: ViewType
    def __init__(self, max_results: _Optional[int] = ..., page_token: _Optional[str] = ..., filter: _Optional[str] = ..., order_by: _Optional[_Iterable[str]] = ..., view_type: _Optional[_Union[ViewType, str]] = ...) -> None: ...

class GetExperiment(_message.Message):
    __slots__ = ("experiment_id",)
    class Response(_message.Message):
        __slots__ = ("experiment",)
        EXPERIMENT_FIELD_NUMBER: _ClassVar[int]
        experiment: Experiment
        def __init__(self, experiment: _Optional[_Union[Experiment, _Mapping]] = ...) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    def __init__(self, experiment_id: _Optional[str] = ...) -> None: ...

class DeleteExperiment(_message.Message):
    __slots__ = ("experiment_id",)
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    def __init__(self, experiment_id: _Optional[str] = ...) -> None: ...

class RestoreExperiment(_message.Message):
    __slots__ = ("experiment_id",)
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    def __init__(self, experiment_id: _Optional[str] = ...) -> None: ...

class UpdateExperiment(_message.Message):
    __slots__ = ("experiment_id", "new_name")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    NEW_NAME_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    new_name: str
    def __init__(self, experiment_id: _Optional[str] = ..., new_name: _Optional[str] = ...) -> None: ...

class CreateRun(_message.Message):
    __slots__ = ("experiment_id", "user_id", "run_name", "start_time", "tags")
    class Response(_message.Message):
        __slots__ = ("run",)
        RUN_FIELD_NUMBER: _ClassVar[int]
        run: Run
        def __init__(self, run: _Optional[_Union[Run, _Mapping]] = ...) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_NAME_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    user_id: str
    run_name: str
    start_time: int
    tags: _containers.RepeatedCompositeFieldContainer[RunTag]
    def __init__(self, experiment_id: _Optional[str] = ..., user_id: _Optional[str] = ..., run_name: _Optional[str] = ..., start_time: _Optional[int] = ..., tags: _Optional[_Iterable[_Union[RunTag, _Mapping]]] = ...) -> None: ...

class UpdateRun(_message.Message):
    __slots__ = ("run_id", "run_uuid", "status", "end_time", "run_name")
    class Response(_message.Message):
        __slots__ = ("run_info",)
        RUN_INFO_FIELD_NUMBER: _ClassVar[int]
        run_info: RunInfo
        def __init__(self, run_info: _Optional[_Union[RunInfo, _Mapping]] = ...) -> None: ...
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_UUID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    RUN_NAME_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    run_uuid: str
    status: RunStatus
    end_time: int
    run_name: str
    def __init__(self, run_id: _Optional[str] = ..., run_uuid: _Optional[str] = ..., status: _Optional[_Union[RunStatus, str]] = ..., end_time: _Optional[int] = ..., run_name: _Optional[str] = ...) -> None: ...

class DeleteRun(_message.Message):
    __slots__ = ("run_id",)
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    def __init__(self, run_id: _Optional[str] = ...) -> None: ...

class RestoreRun(_message.Message):
    __slots__ = ("run_id",)
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    def __init__(self, run_id: _Optional[str] = ...) -> None: ...

class LogMetric(_message.Message):
    __slots__ = ("run_id", "run_uuid", "key", "value", "timestamp", "step", "model_id", "dataset_name", "dataset_digest")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_UUID_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    DATASET_NAME_FIELD_NUMBER: _ClassVar[int]
    DATASET_DIGEST_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    run_uuid: str
    key: str
    value: float
    timestamp: int
    step: int
    model_id: str
    dataset_name: str
    dataset_digest: str
    def __init__(self, run_id: _Optional[str] = ..., run_uuid: _Optional[str] = ..., key: _Optional[str] = ..., value: _Optional[float] = ..., timestamp: _Optional[int] = ..., step: _Optional[int] = ..., model_id: _Optional[str] = ..., dataset_name: _Optional[str] = ..., dataset_digest: _Optional[str] = ...) -> None: ...

class LogParam(_message.Message):
    __slots__ = ("run_id", "run_uuid", "key", "value")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_UUID_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    run_uuid: str
    key: str
    value: str
    def __init__(self, run_id: _Optional[str] = ..., run_uuid: _Optional[str] = ..., key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class SetExperimentTag(_message.Message):
    __slots__ = ("experiment_id", "key", "value")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    key: str
    value: str
    def __init__(self, experiment_id: _Optional[str] = ..., key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class DeleteExperimentTag(_message.Message):
    __slots__ = ("experiment_id", "key")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    key: str
    def __init__(self, experiment_id: _Optional[str] = ..., key: _Optional[str] = ...) -> None: ...

class SetTag(_message.Message):
    __slots__ = ("run_id", "run_uuid", "key", "value")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_UUID_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    run_uuid: str
    key: str
    value: str
    def __init__(self, run_id: _Optional[str] = ..., run_uuid: _Optional[str] = ..., key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class DeleteTag(_message.Message):
    __slots__ = ("run_id", "key")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    key: str
    def __init__(self, run_id: _Optional[str] = ..., key: _Optional[str] = ...) -> None: ...

class GetRun(_message.Message):
    __slots__ = ("run_id", "run_uuid")
    class Response(_message.Message):
        __slots__ = ("run",)
        RUN_FIELD_NUMBER: _ClassVar[int]
        run: Run
        def __init__(self, run: _Optional[_Union[Run, _Mapping]] = ...) -> None: ...
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_UUID_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    run_uuid: str
    def __init__(self, run_id: _Optional[str] = ..., run_uuid: _Optional[str] = ...) -> None: ...

class SearchRuns(_message.Message):
    __slots__ = ("experiment_ids", "filter", "run_view_type", "max_results", "order_by", "page_token")
    class Response(_message.Message):
        __slots__ = ("runs", "next_page_token")
        RUNS_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        runs: _containers.RepeatedCompositeFieldContainer[Run]
        next_page_token: str
        def __init__(self, runs: _Optional[_Iterable[_Union[Run, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    EXPERIMENT_IDS_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    RUN_VIEW_TYPE_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    ORDER_BY_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    experiment_ids: _containers.RepeatedScalarFieldContainer[str]
    filter: str
    run_view_type: ViewType
    max_results: int
    order_by: _containers.RepeatedScalarFieldContainer[str]
    page_token: str
    def __init__(self, experiment_ids: _Optional[_Iterable[str]] = ..., filter: _Optional[str] = ..., run_view_type: _Optional[_Union[ViewType, str]] = ..., max_results: _Optional[int] = ..., order_by: _Optional[_Iterable[str]] = ..., page_token: _Optional[str] = ...) -> None: ...

class ListArtifacts(_message.Message):
    __slots__ = ("run_id", "run_uuid", "path", "page_token")
    class Response(_message.Message):
        __slots__ = ("root_uri", "files", "next_page_token")
        ROOT_URI_FIELD_NUMBER: _ClassVar[int]
        FILES_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        root_uri: str
        files: _containers.RepeatedCompositeFieldContainer[FileInfo]
        next_page_token: str
        def __init__(self, root_uri: _Optional[str] = ..., files: _Optional[_Iterable[_Union[FileInfo, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_UUID_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    run_uuid: str
    path: str
    page_token: str
    def __init__(self, run_id: _Optional[str] = ..., run_uuid: _Optional[str] = ..., path: _Optional[str] = ..., page_token: _Optional[str] = ...) -> None: ...

class FileInfo(_message.Message):
    __slots__ = ("path", "is_dir", "file_size")
    PATH_FIELD_NUMBER: _ClassVar[int]
    IS_DIR_FIELD_NUMBER: _ClassVar[int]
    FILE_SIZE_FIELD_NUMBER: _ClassVar[int]
    path: str
    is_dir: bool
    file_size: int
    def __init__(self, path: _Optional[str] = ..., is_dir: bool = ..., file_size: _Optional[int] = ...) -> None: ...

class GetMetricHistory(_message.Message):
    __slots__ = ("run_id", "run_uuid", "metric_key", "page_token", "max_results")
    class Response(_message.Message):
        __slots__ = ("metrics", "next_page_token")
        METRICS_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        metrics: _containers.RepeatedCompositeFieldContainer[Metric]
        next_page_token: str
        def __init__(self, metrics: _Optional[_Iterable[_Union[Metric, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_UUID_FIELD_NUMBER: _ClassVar[int]
    METRIC_KEY_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    run_uuid: str
    metric_key: str
    page_token: str
    max_results: int
    def __init__(self, run_id: _Optional[str] = ..., run_uuid: _Optional[str] = ..., metric_key: _Optional[str] = ..., page_token: _Optional[str] = ..., max_results: _Optional[int] = ...) -> None: ...

class MetricWithRunId(_message.Message):
    __slots__ = ("key", "value", "timestamp", "step", "run_id")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: float
    timestamp: int
    step: int
    run_id: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ..., timestamp: _Optional[int] = ..., step: _Optional[int] = ..., run_id: _Optional[str] = ...) -> None: ...

class GetMetricHistoryBulkInterval(_message.Message):
    __slots__ = ("run_ids", "metric_key", "start_step", "end_step", "max_results")
    class Response(_message.Message):
        __slots__ = ("metrics",)
        METRICS_FIELD_NUMBER: _ClassVar[int]
        metrics: _containers.RepeatedCompositeFieldContainer[MetricWithRunId]
        def __init__(self, metrics: _Optional[_Iterable[_Union[MetricWithRunId, _Mapping]]] = ...) -> None: ...
    RUN_IDS_FIELD_NUMBER: _ClassVar[int]
    METRIC_KEY_FIELD_NUMBER: _ClassVar[int]
    START_STEP_FIELD_NUMBER: _ClassVar[int]
    END_STEP_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    run_ids: _containers.RepeatedScalarFieldContainer[str]
    metric_key: str
    start_step: int
    end_step: int
    max_results: int
    def __init__(self, run_ids: _Optional[_Iterable[str]] = ..., metric_key: _Optional[str] = ..., start_step: _Optional[int] = ..., end_step: _Optional[int] = ..., max_results: _Optional[int] = ...) -> None: ...

class LogBatch(_message.Message):
    __slots__ = ("run_id", "metrics", "params", "tags")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    metrics: _containers.RepeatedCompositeFieldContainer[Metric]
    params: _containers.RepeatedCompositeFieldContainer[Param]
    tags: _containers.RepeatedCompositeFieldContainer[RunTag]
    def __init__(self, run_id: _Optional[str] = ..., metrics: _Optional[_Iterable[_Union[Metric, _Mapping]]] = ..., params: _Optional[_Iterable[_Union[Param, _Mapping]]] = ..., tags: _Optional[_Iterable[_Union[RunTag, _Mapping]]] = ...) -> None: ...

class LogModel(_message.Message):
    __slots__ = ("run_id", "model_json")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_JSON_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    model_json: str
    def __init__(self, run_id: _Optional[str] = ..., model_json: _Optional[str] = ...) -> None: ...

class LogInputs(_message.Message):
    __slots__ = ("run_id", "datasets", "models")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    DATASETS_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    datasets: _containers.RepeatedCompositeFieldContainer[DatasetInput]
    models: _containers.RepeatedCompositeFieldContainer[ModelInput]
    def __init__(self, run_id: _Optional[str] = ..., datasets: _Optional[_Iterable[_Union[DatasetInput, _Mapping]]] = ..., models: _Optional[_Iterable[_Union[ModelInput, _Mapping]]] = ...) -> None: ...

class LogOutputs(_message.Message):
    __slots__ = ("run_id", "models")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    models: _containers.RepeatedCompositeFieldContainer[ModelOutput]
    def __init__(self, run_id: _Optional[str] = ..., models: _Optional[_Iterable[_Union[ModelOutput, _Mapping]]] = ...) -> None: ...

class GetExperimentByName(_message.Message):
    __slots__ = ("experiment_name",)
    class Response(_message.Message):
        __slots__ = ("experiment",)
        EXPERIMENT_FIELD_NUMBER: _ClassVar[int]
        experiment: Experiment
        def __init__(self, experiment: _Optional[_Union[Experiment, _Mapping]] = ...) -> None: ...
    EXPERIMENT_NAME_FIELD_NUMBER: _ClassVar[int]
    experiment_name: str
    def __init__(self, experiment_name: _Optional[str] = ...) -> None: ...

class CreateAssessment(_message.Message):
    __slots__ = ("assessment",)
    class Response(_message.Message):
        __slots__ = ("assessment",)
        ASSESSMENT_FIELD_NUMBER: _ClassVar[int]
        assessment: _assessments_pb2.Assessment
        def __init__(self, assessment: _Optional[_Union[_assessments_pb2.Assessment, _Mapping]] = ...) -> None: ...
    ASSESSMENT_FIELD_NUMBER: _ClassVar[int]
    assessment: _assessments_pb2.Assessment
    def __init__(self, assessment: _Optional[_Union[_assessments_pb2.Assessment, _Mapping]] = ...) -> None: ...

class UpdateAssessment(_message.Message):
    __slots__ = ("assessment", "update_mask")
    class Response(_message.Message):
        __slots__ = ("assessment",)
        ASSESSMENT_FIELD_NUMBER: _ClassVar[int]
        assessment: _assessments_pb2.Assessment
        def __init__(self, assessment: _Optional[_Union[_assessments_pb2.Assessment, _Mapping]] = ...) -> None: ...
    ASSESSMENT_FIELD_NUMBER: _ClassVar[int]
    UPDATE_MASK_FIELD_NUMBER: _ClassVar[int]
    assessment: _assessments_pb2.Assessment
    update_mask: _field_mask_pb2.FieldMask
    def __init__(self, assessment: _Optional[_Union[_assessments_pb2.Assessment, _Mapping]] = ..., update_mask: _Optional[_Union[_field_mask_pb2.FieldMask, _Mapping]] = ...) -> None: ...

class DeleteAssessment(_message.Message):
    __slots__ = ("trace_id", "assessment_id")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    ASSESSMENT_ID_FIELD_NUMBER: _ClassVar[int]
    trace_id: str
    assessment_id: str
    def __init__(self, trace_id: _Optional[str] = ..., assessment_id: _Optional[str] = ...) -> None: ...

class GetAssessmentRequest(_message.Message):
    __slots__ = ("trace_id", "assessment_id")
    class Response(_message.Message):
        __slots__ = ("assessment",)
        ASSESSMENT_FIELD_NUMBER: _ClassVar[int]
        assessment: _assessments_pb2.Assessment
        def __init__(self, assessment: _Optional[_Union[_assessments_pb2.Assessment, _Mapping]] = ...) -> None: ...
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    ASSESSMENT_ID_FIELD_NUMBER: _ClassVar[int]
    trace_id: str
    assessment_id: str
    def __init__(self, trace_id: _Optional[str] = ..., assessment_id: _Optional[str] = ...) -> None: ...

class TraceInfo(_message.Message):
    __slots__ = ("request_id", "experiment_id", "timestamp_ms", "execution_time_ms", "status", "request_metadata", "tags")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    REQUEST_METADATA_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    experiment_id: str
    timestamp_ms: int
    execution_time_ms: int
    status: TraceStatus
    request_metadata: _containers.RepeatedCompositeFieldContainer[TraceRequestMetadata]
    tags: _containers.RepeatedCompositeFieldContainer[TraceTag]
    def __init__(self, request_id: _Optional[str] = ..., experiment_id: _Optional[str] = ..., timestamp_ms: _Optional[int] = ..., execution_time_ms: _Optional[int] = ..., status: _Optional[_Union[TraceStatus, str]] = ..., request_metadata: _Optional[_Iterable[_Union[TraceRequestMetadata, _Mapping]]] = ..., tags: _Optional[_Iterable[_Union[TraceTag, _Mapping]]] = ...) -> None: ...

class TraceRequestMetadata(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class TraceTag(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class StartTrace(_message.Message):
    __slots__ = ("experiment_id", "timestamp_ms", "request_metadata", "tags")
    class Response(_message.Message):
        __slots__ = ("trace_info",)
        TRACE_INFO_FIELD_NUMBER: _ClassVar[int]
        trace_info: TraceInfo
        def __init__(self, trace_info: _Optional[_Union[TraceInfo, _Mapping]] = ...) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    REQUEST_METADATA_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    timestamp_ms: int
    request_metadata: _containers.RepeatedCompositeFieldContainer[TraceRequestMetadata]
    tags: _containers.RepeatedCompositeFieldContainer[TraceTag]
    def __init__(self, experiment_id: _Optional[str] = ..., timestamp_ms: _Optional[int] = ..., request_metadata: _Optional[_Iterable[_Union[TraceRequestMetadata, _Mapping]]] = ..., tags: _Optional[_Iterable[_Union[TraceTag, _Mapping]]] = ...) -> None: ...

class EndTrace(_message.Message):
    __slots__ = ("request_id", "timestamp_ms", "status", "request_metadata", "tags")
    class Response(_message.Message):
        __slots__ = ("trace_info",)
        TRACE_INFO_FIELD_NUMBER: _ClassVar[int]
        trace_info: TraceInfo
        def __init__(self, trace_info: _Optional[_Union[TraceInfo, _Mapping]] = ...) -> None: ...
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    REQUEST_METADATA_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    timestamp_ms: int
    status: TraceStatus
    request_metadata: _containers.RepeatedCompositeFieldContainer[TraceRequestMetadata]
    tags: _containers.RepeatedCompositeFieldContainer[TraceTag]
    def __init__(self, request_id: _Optional[str] = ..., timestamp_ms: _Optional[int] = ..., status: _Optional[_Union[TraceStatus, str]] = ..., request_metadata: _Optional[_Iterable[_Union[TraceRequestMetadata, _Mapping]]] = ..., tags: _Optional[_Iterable[_Union[TraceTag, _Mapping]]] = ...) -> None: ...

class GetTraceInfo(_message.Message):
    __slots__ = ("request_id",)
    class Response(_message.Message):
        __slots__ = ("trace_info",)
        TRACE_INFO_FIELD_NUMBER: _ClassVar[int]
        trace_info: TraceInfo
        def __init__(self, trace_info: _Optional[_Union[TraceInfo, _Mapping]] = ...) -> None: ...
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    def __init__(self, request_id: _Optional[str] = ...) -> None: ...

class GetTraceInfoV3(_message.Message):
    __slots__ = ("trace_id",)
    class Response(_message.Message):
        __slots__ = ("trace",)
        TRACE_FIELD_NUMBER: _ClassVar[int]
        trace: Trace
        def __init__(self, trace: _Optional[_Union[Trace, _Mapping]] = ...) -> None: ...
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    trace_id: str
    def __init__(self, trace_id: _Optional[str] = ...) -> None: ...

class SearchTraces(_message.Message):
    __slots__ = ("experiment_ids", "filter", "max_results", "order_by", "page_token")
    class Response(_message.Message):
        __slots__ = ("traces", "next_page_token")
        TRACES_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        traces: _containers.RepeatedCompositeFieldContainer[TraceInfo]
        next_page_token: str
        def __init__(self, traces: _Optional[_Iterable[_Union[TraceInfo, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    EXPERIMENT_IDS_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    ORDER_BY_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    experiment_ids: _containers.RepeatedScalarFieldContainer[str]
    filter: str
    max_results: int
    order_by: _containers.RepeatedScalarFieldContainer[str]
    page_token: str
    def __init__(self, experiment_ids: _Optional[_Iterable[str]] = ..., filter: _Optional[str] = ..., max_results: _Optional[int] = ..., order_by: _Optional[_Iterable[str]] = ..., page_token: _Optional[str] = ...) -> None: ...

class SearchUnifiedTraces(_message.Message):
    __slots__ = ("model_id", "sql_warehouse_id", "experiment_ids", "filter", "max_results", "order_by", "page_token")
    class Response(_message.Message):
        __slots__ = ("traces", "next_page_token")
        TRACES_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        traces: _containers.RepeatedCompositeFieldContainer[TraceInfo]
        next_page_token: str
        def __init__(self, traces: _Optional[_Iterable[_Union[TraceInfo, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    SQL_WAREHOUSE_ID_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_IDS_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    ORDER_BY_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    sql_warehouse_id: str
    experiment_ids: _containers.RepeatedScalarFieldContainer[str]
    filter: str
    max_results: int
    order_by: _containers.RepeatedScalarFieldContainer[str]
    page_token: str
    def __init__(self, model_id: _Optional[str] = ..., sql_warehouse_id: _Optional[str] = ..., experiment_ids: _Optional[_Iterable[str]] = ..., filter: _Optional[str] = ..., max_results: _Optional[int] = ..., order_by: _Optional[_Iterable[str]] = ..., page_token: _Optional[str] = ...) -> None: ...

class GetOnlineTraceDetails(_message.Message):
    __slots__ = ("trace_id", "sql_warehouse_id", "source_inference_table", "source_databricks_request_id")
    class Response(_message.Message):
        __slots__ = ("trace_data",)
        TRACE_DATA_FIELD_NUMBER: _ClassVar[int]
        trace_data: str
        def __init__(self, trace_data: _Optional[str] = ...) -> None: ...
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    SQL_WAREHOUSE_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_INFERENCE_TABLE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_DATABRICKS_REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    trace_id: str
    sql_warehouse_id: str
    source_inference_table: str
    source_databricks_request_id: str
    def __init__(self, trace_id: _Optional[str] = ..., sql_warehouse_id: _Optional[str] = ..., source_inference_table: _Optional[str] = ..., source_databricks_request_id: _Optional[str] = ...) -> None: ...

class DeleteTraces(_message.Message):
    __slots__ = ("experiment_id", "max_timestamp_millis", "max_traces", "request_ids")
    class Response(_message.Message):
        __slots__ = ("traces_deleted",)
        TRACES_DELETED_FIELD_NUMBER: _ClassVar[int]
        traces_deleted: int
        def __init__(self, traces_deleted: _Optional[int] = ...) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    MAX_TIMESTAMP_MILLIS_FIELD_NUMBER: _ClassVar[int]
    MAX_TRACES_FIELD_NUMBER: _ClassVar[int]
    REQUEST_IDS_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    max_timestamp_millis: int
    max_traces: int
    request_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, experiment_id: _Optional[str] = ..., max_timestamp_millis: _Optional[int] = ..., max_traces: _Optional[int] = ..., request_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class DeleteTracesV3(_message.Message):
    __slots__ = ("experiment_id", "max_timestamp_millis", "max_traces", "request_ids")
    class Response(_message.Message):
        __slots__ = ("traces_deleted",)
        TRACES_DELETED_FIELD_NUMBER: _ClassVar[int]
        traces_deleted: int
        def __init__(self, traces_deleted: _Optional[int] = ...) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    MAX_TIMESTAMP_MILLIS_FIELD_NUMBER: _ClassVar[int]
    MAX_TRACES_FIELD_NUMBER: _ClassVar[int]
    REQUEST_IDS_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    max_timestamp_millis: int
    max_traces: int
    request_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, experiment_id: _Optional[str] = ..., max_timestamp_millis: _Optional[int] = ..., max_traces: _Optional[int] = ..., request_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class CalculateTraceFilterCorrelation(_message.Message):
    __slots__ = ("experiment_ids", "filter_string1", "filter_string2", "base_filter")
    class Response(_message.Message):
        __slots__ = ("npmi", "npmi_smoothed", "filter1_count", "filter2_count", "joint_count", "total_count")
        NPMI_FIELD_NUMBER: _ClassVar[int]
        NPMI_SMOOTHED_FIELD_NUMBER: _ClassVar[int]
        FILTER1_COUNT_FIELD_NUMBER: _ClassVar[int]
        FILTER2_COUNT_FIELD_NUMBER: _ClassVar[int]
        JOINT_COUNT_FIELD_NUMBER: _ClassVar[int]
        TOTAL_COUNT_FIELD_NUMBER: _ClassVar[int]
        npmi: float
        npmi_smoothed: float
        filter1_count: int
        filter2_count: int
        joint_count: int
        total_count: int
        def __init__(self, npmi: _Optional[float] = ..., npmi_smoothed: _Optional[float] = ..., filter1_count: _Optional[int] = ..., filter2_count: _Optional[int] = ..., joint_count: _Optional[int] = ..., total_count: _Optional[int] = ...) -> None: ...
    EXPERIMENT_IDS_FIELD_NUMBER: _ClassVar[int]
    FILTER_STRING1_FIELD_NUMBER: _ClassVar[int]
    FILTER_STRING2_FIELD_NUMBER: _ClassVar[int]
    BASE_FILTER_FIELD_NUMBER: _ClassVar[int]
    experiment_ids: _containers.RepeatedScalarFieldContainer[str]
    filter_string1: str
    filter_string2: str
    base_filter: str
    def __init__(self, experiment_ids: _Optional[_Iterable[str]] = ..., filter_string1: _Optional[str] = ..., filter_string2: _Optional[str] = ..., base_filter: _Optional[str] = ...) -> None: ...

class SetTraceTag(_message.Message):
    __slots__ = ("request_id", "key", "value")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    key: str
    value: str
    def __init__(self, request_id: _Optional[str] = ..., key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class SetTraceTagV3(_message.Message):
    __slots__ = ("trace_id", "key", "value")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    trace_id: str
    key: str
    value: str
    def __init__(self, trace_id: _Optional[str] = ..., key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class DeleteTraceTag(_message.Message):
    __slots__ = ("trace_id", "key")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    trace_id: str
    key: str
    def __init__(self, trace_id: _Optional[str] = ..., key: _Optional[str] = ...) -> None: ...

class DeleteTraceTagV3(_message.Message):
    __slots__ = ("request_id", "key")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    key: str
    def __init__(self, request_id: _Optional[str] = ..., key: _Optional[str] = ...) -> None: ...

class Trace(_message.Message):
    __slots__ = ("trace_info",)
    TRACE_INFO_FIELD_NUMBER: _ClassVar[int]
    trace_info: TraceInfoV3
    def __init__(self, trace_info: _Optional[_Union[TraceInfoV3, _Mapping]] = ...) -> None: ...

class TraceLocation(_message.Message):
    __slots__ = ("type", "mlflow_experiment", "inference_table")
    class TraceLocationType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        TRACE_LOCATION_TYPE_UNSPECIFIED: _ClassVar[TraceLocation.TraceLocationType]
        MLFLOW_EXPERIMENT: _ClassVar[TraceLocation.TraceLocationType]
        INFERENCE_TABLE: _ClassVar[TraceLocation.TraceLocationType]
    TRACE_LOCATION_TYPE_UNSPECIFIED: TraceLocation.TraceLocationType
    MLFLOW_EXPERIMENT: TraceLocation.TraceLocationType
    INFERENCE_TABLE: TraceLocation.TraceLocationType
    class MlflowExperimentLocation(_message.Message):
        __slots__ = ("experiment_id",)
        EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
        experiment_id: str
        def __init__(self, experiment_id: _Optional[str] = ...) -> None: ...
    class InferenceTableLocation(_message.Message):
        __slots__ = ("full_table_name",)
        FULL_TABLE_NAME_FIELD_NUMBER: _ClassVar[int]
        full_table_name: str
        def __init__(self, full_table_name: _Optional[str] = ...) -> None: ...
    TYPE_FIELD_NUMBER: _ClassVar[int]
    MLFLOW_EXPERIMENT_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_TABLE_FIELD_NUMBER: _ClassVar[int]
    type: TraceLocation.TraceLocationType
    mlflow_experiment: TraceLocation.MlflowExperimentLocation
    inference_table: TraceLocation.InferenceTableLocation
    def __init__(self, type: _Optional[_Union[TraceLocation.TraceLocationType, str]] = ..., mlflow_experiment: _Optional[_Union[TraceLocation.MlflowExperimentLocation, _Mapping]] = ..., inference_table: _Optional[_Union[TraceLocation.InferenceTableLocation, _Mapping]] = ...) -> None: ...

class TraceInfoV3(_message.Message):
    __slots__ = ("trace_id", "client_request_id", "trace_location", "request", "response", "request_preview", "response_preview", "request_time", "execution_duration", "state", "trace_metadata", "assessments", "tags")
    class State(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STATE_UNSPECIFIED: _ClassVar[TraceInfoV3.State]
        OK: _ClassVar[TraceInfoV3.State]
        ERROR: _ClassVar[TraceInfoV3.State]
        IN_PROGRESS: _ClassVar[TraceInfoV3.State]
    STATE_UNSPECIFIED: TraceInfoV3.State
    OK: TraceInfoV3.State
    ERROR: TraceInfoV3.State
    IN_PROGRESS: TraceInfoV3.State
    class TraceMetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class TagsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    CLIENT_REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TRACE_LOCATION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_PREVIEW_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_PREVIEW_FIELD_NUMBER: _ClassVar[int]
    REQUEST_TIME_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_DURATION_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    TRACE_METADATA_FIELD_NUMBER: _ClassVar[int]
    ASSESSMENTS_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    trace_id: str
    client_request_id: str
    trace_location: TraceLocation
    request: str
    response: str
    request_preview: str
    response_preview: str
    request_time: _timestamp_pb2.Timestamp
    execution_duration: _duration_pb2.Duration
    state: TraceInfoV3.State
    trace_metadata: _containers.ScalarMap[str, str]
    assessments: _containers.RepeatedCompositeFieldContainer[_assessments_pb2.Assessment]
    tags: _containers.ScalarMap[str, str]
    def __init__(self, trace_id: _Optional[str] = ..., client_request_id: _Optional[str] = ..., trace_location: _Optional[_Union[TraceLocation, _Mapping]] = ..., request: _Optional[str] = ..., response: _Optional[str] = ..., request_preview: _Optional[str] = ..., response_preview: _Optional[str] = ..., request_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., execution_duration: _Optional[_Union[_duration_pb2.Duration, _Mapping]] = ..., state: _Optional[_Union[TraceInfoV3.State, str]] = ..., trace_metadata: _Optional[_Mapping[str, str]] = ..., assessments: _Optional[_Iterable[_Union[_assessments_pb2.Assessment, _Mapping]]] = ..., tags: _Optional[_Mapping[str, str]] = ...) -> None: ...

class StartTraceV3(_message.Message):
    __slots__ = ("trace",)
    class Response(_message.Message):
        __slots__ = ("trace",)
        TRACE_FIELD_NUMBER: _ClassVar[int]
        trace: Trace
        def __init__(self, trace: _Optional[_Union[Trace, _Mapping]] = ...) -> None: ...
    TRACE_FIELD_NUMBER: _ClassVar[int]
    trace: Trace
    def __init__(self, trace: _Optional[_Union[Trace, _Mapping]] = ...) -> None: ...

class LinkTracesToRun(_message.Message):
    __slots__ = ("trace_ids", "run_id")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    TRACE_IDS_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    trace_ids: _containers.RepeatedScalarFieldContainer[str]
    run_id: str
    def __init__(self, trace_ids: _Optional[_Iterable[str]] = ..., run_id: _Optional[str] = ...) -> None: ...

class DatasetSummary(_message.Message):
    __slots__ = ("experiment_id", "name", "digest", "context")
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    name: str
    digest: str
    context: str
    def __init__(self, experiment_id: _Optional[str] = ..., name: _Optional[str] = ..., digest: _Optional[str] = ..., context: _Optional[str] = ...) -> None: ...

class SearchDatasets(_message.Message):
    __slots__ = ("experiment_ids",)
    class Response(_message.Message):
        __slots__ = ("dataset_summaries",)
        DATASET_SUMMARIES_FIELD_NUMBER: _ClassVar[int]
        dataset_summaries: _containers.RepeatedCompositeFieldContainer[DatasetSummary]
        def __init__(self, dataset_summaries: _Optional[_Iterable[_Union[DatasetSummary, _Mapping]]] = ...) -> None: ...
    EXPERIMENT_IDS_FIELD_NUMBER: _ClassVar[int]
    experiment_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, experiment_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class CreateLoggedModel(_message.Message):
    __slots__ = ("experiment_id", "name", "model_type", "source_run_id", "params", "tags")
    class Response(_message.Message):
        __slots__ = ("model",)
        MODEL_FIELD_NUMBER: _ClassVar[int]
        model: LoggedModel
        def __init__(self, model: _Optional[_Union[LoggedModel, _Mapping]] = ...) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_RUN_ID_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    name: str
    model_type: str
    source_run_id: str
    params: _containers.RepeatedCompositeFieldContainer[LoggedModelParameter]
    tags: _containers.RepeatedCompositeFieldContainer[LoggedModelTag]
    def __init__(self, experiment_id: _Optional[str] = ..., name: _Optional[str] = ..., model_type: _Optional[str] = ..., source_run_id: _Optional[str] = ..., params: _Optional[_Iterable[_Union[LoggedModelParameter, _Mapping]]] = ..., tags: _Optional[_Iterable[_Union[LoggedModelTag, _Mapping]]] = ...) -> None: ...

class FinalizeLoggedModel(_message.Message):
    __slots__ = ("model_id", "status")
    class Response(_message.Message):
        __slots__ = ("model",)
        MODEL_FIELD_NUMBER: _ClassVar[int]
        model: LoggedModel
        def __init__(self, model: _Optional[_Union[LoggedModel, _Mapping]] = ...) -> None: ...
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    status: LoggedModelStatus
    def __init__(self, model_id: _Optional[str] = ..., status: _Optional[_Union[LoggedModelStatus, str]] = ...) -> None: ...

class GetLoggedModel(_message.Message):
    __slots__ = ("model_id",)
    class Response(_message.Message):
        __slots__ = ("model",)
        MODEL_FIELD_NUMBER: _ClassVar[int]
        model: LoggedModel
        def __init__(self, model: _Optional[_Union[LoggedModel, _Mapping]] = ...) -> None: ...
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    def __init__(self, model_id: _Optional[str] = ...) -> None: ...

class DeleteLoggedModel(_message.Message):
    __slots__ = ("model_id",)
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    def __init__(self, model_id: _Optional[str] = ...) -> None: ...

class SearchLoggedModels(_message.Message):
    __slots__ = ("experiment_ids", "filter", "datasets", "max_results", "order_by", "page_token")
    class Dataset(_message.Message):
        __slots__ = ("dataset_name", "dataset_digest")
        DATASET_NAME_FIELD_NUMBER: _ClassVar[int]
        DATASET_DIGEST_FIELD_NUMBER: _ClassVar[int]
        dataset_name: str
        dataset_digest: str
        def __init__(self, dataset_name: _Optional[str] = ..., dataset_digest: _Optional[str] = ...) -> None: ...
    class OrderBy(_message.Message):
        __slots__ = ("field_name", "ascending", "dataset_name", "dataset_digest")
        FIELD_NAME_FIELD_NUMBER: _ClassVar[int]
        ASCENDING_FIELD_NUMBER: _ClassVar[int]
        DATASET_NAME_FIELD_NUMBER: _ClassVar[int]
        DATASET_DIGEST_FIELD_NUMBER: _ClassVar[int]
        field_name: str
        ascending: bool
        dataset_name: str
        dataset_digest: str
        def __init__(self, field_name: _Optional[str] = ..., ascending: bool = ..., dataset_name: _Optional[str] = ..., dataset_digest: _Optional[str] = ...) -> None: ...
    class Response(_message.Message):
        __slots__ = ("models", "next_page_token")
        MODELS_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        models: _containers.RepeatedCompositeFieldContainer[LoggedModel]
        next_page_token: str
        def __init__(self, models: _Optional[_Iterable[_Union[LoggedModel, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    EXPERIMENT_IDS_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    DATASETS_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    ORDER_BY_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    experiment_ids: _containers.RepeatedScalarFieldContainer[str]
    filter: str
    datasets: _containers.RepeatedCompositeFieldContainer[SearchLoggedModels.Dataset]
    max_results: int
    order_by: _containers.RepeatedCompositeFieldContainer[SearchLoggedModels.OrderBy]
    page_token: str
    def __init__(self, experiment_ids: _Optional[_Iterable[str]] = ..., filter: _Optional[str] = ..., datasets: _Optional[_Iterable[_Union[SearchLoggedModels.Dataset, _Mapping]]] = ..., max_results: _Optional[int] = ..., order_by: _Optional[_Iterable[_Union[SearchLoggedModels.OrderBy, _Mapping]]] = ..., page_token: _Optional[str] = ...) -> None: ...

class SetLoggedModelTags(_message.Message):
    __slots__ = ("model_id", "tags")
    class Response(_message.Message):
        __slots__ = ("model",)
        MODEL_FIELD_NUMBER: _ClassVar[int]
        model: LoggedModel
        def __init__(self, model: _Optional[_Union[LoggedModel, _Mapping]] = ...) -> None: ...
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    tags: _containers.RepeatedCompositeFieldContainer[LoggedModelTag]
    def __init__(self, model_id: _Optional[str] = ..., tags: _Optional[_Iterable[_Union[LoggedModelTag, _Mapping]]] = ...) -> None: ...

class DeleteLoggedModelTag(_message.Message):
    __slots__ = ("model_id", "tag_key")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    TAG_KEY_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    tag_key: str
    def __init__(self, model_id: _Optional[str] = ..., tag_key: _Optional[str] = ...) -> None: ...

class ListLoggedModelArtifacts(_message.Message):
    __slots__ = ("model_id", "artifact_directory_path", "page_token")
    class Response(_message.Message):
        __slots__ = ("root_uri", "files", "next_page_token")
        ROOT_URI_FIELD_NUMBER: _ClassVar[int]
        FILES_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        root_uri: str
        files: _containers.RepeatedCompositeFieldContainer[FileInfo]
        next_page_token: str
        def __init__(self, root_uri: _Optional[str] = ..., files: _Optional[_Iterable[_Union[FileInfo, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_DIRECTORY_PATH_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    artifact_directory_path: str
    page_token: str
    def __init__(self, model_id: _Optional[str] = ..., artifact_directory_path: _Optional[str] = ..., page_token: _Optional[str] = ...) -> None: ...

class LogLoggedModelParamsRequest(_message.Message):
    __slots__ = ("model_id", "params")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    params: _containers.RepeatedCompositeFieldContainer[LoggedModelParameter]
    def __init__(self, model_id: _Optional[str] = ..., params: _Optional[_Iterable[_Union[LoggedModelParameter, _Mapping]]] = ...) -> None: ...

class LoggedModel(_message.Message):
    __slots__ = ("info", "data")
    INFO_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    info: LoggedModelInfo
    data: LoggedModelData
    def __init__(self, info: _Optional[_Union[LoggedModelInfo, _Mapping]] = ..., data: _Optional[_Union[LoggedModelData, _Mapping]] = ...) -> None: ...

class LoggedModelInfo(_message.Message):
    __slots__ = ("model_id", "experiment_id", "name", "creation_timestamp_ms", "last_updated_timestamp_ms", "artifact_uri", "status", "creator_id", "model_type", "source_run_id", "status_message", "tags", "registrations")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    CREATION_TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    LAST_UPDATED_TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_URI_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CREATOR_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_RUN_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    REGISTRATIONS_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    experiment_id: str
    name: str
    creation_timestamp_ms: int
    last_updated_timestamp_ms: int
    artifact_uri: str
    status: LoggedModelStatus
    creator_id: int
    model_type: str
    source_run_id: str
    status_message: str
    tags: _containers.RepeatedCompositeFieldContainer[LoggedModelTag]
    registrations: _containers.RepeatedCompositeFieldContainer[LoggedModelRegistrationInfo]
    def __init__(self, model_id: _Optional[str] = ..., experiment_id: _Optional[str] = ..., name: _Optional[str] = ..., creation_timestamp_ms: _Optional[int] = ..., last_updated_timestamp_ms: _Optional[int] = ..., artifact_uri: _Optional[str] = ..., status: _Optional[_Union[LoggedModelStatus, str]] = ..., creator_id: _Optional[int] = ..., model_type: _Optional[str] = ..., source_run_id: _Optional[str] = ..., status_message: _Optional[str] = ..., tags: _Optional[_Iterable[_Union[LoggedModelTag, _Mapping]]] = ..., registrations: _Optional[_Iterable[_Union[LoggedModelRegistrationInfo, _Mapping]]] = ...) -> None: ...

class LoggedModelTag(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class LoggedModelRegistrationInfo(_message.Message):
    __slots__ = ("name", "version")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class LoggedModelData(_message.Message):
    __slots__ = ("params", "metrics")
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    params: _containers.RepeatedCompositeFieldContainer[LoggedModelParameter]
    metrics: _containers.RepeatedCompositeFieldContainer[Metric]
    def __init__(self, params: _Optional[_Iterable[_Union[LoggedModelParameter, _Mapping]]] = ..., metrics: _Optional[_Iterable[_Union[Metric, _Mapping]]] = ...) -> None: ...

class LoggedModelParameter(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class SearchTracesV3(_message.Message):
    __slots__ = ("locations", "filter", "max_results", "order_by", "page_token")
    class Response(_message.Message):
        __slots__ = ("traces", "next_page_token")
        TRACES_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        traces: _containers.RepeatedCompositeFieldContainer[TraceInfoV3]
        next_page_token: str
        def __init__(self, traces: _Optional[_Iterable[_Union[TraceInfoV3, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    LOCATIONS_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    ORDER_BY_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    locations: _containers.RepeatedCompositeFieldContainer[TraceLocation]
    filter: str
    max_results: int
    order_by: _containers.RepeatedScalarFieldContainer[str]
    page_token: str
    def __init__(self, locations: _Optional[_Iterable[_Union[TraceLocation, _Mapping]]] = ..., filter: _Optional[str] = ..., max_results: _Optional[int] = ..., order_by: _Optional[_Iterable[str]] = ..., page_token: _Optional[str] = ...) -> None: ...

class CreateDataset(_message.Message):
    __slots__ = ("name", "experiment_ids", "source_type", "source", "schema", "profile", "created_by", "tags")
    class Response(_message.Message):
        __slots__ = ("dataset",)
        DATASET_FIELD_NUMBER: _ClassVar[int]
        dataset: _datasets_pb2.Dataset
        def __init__(self, dataset: _Optional[_Union[_datasets_pb2.Dataset, _Mapping]] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_IDS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    PROFILE_FIELD_NUMBER: _ClassVar[int]
    CREATED_BY_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    name: str
    experiment_ids: _containers.RepeatedScalarFieldContainer[str]
    source_type: _datasets_pb2.DatasetRecordSource.SourceType
    source: str
    schema: str
    profile: str
    created_by: str
    tags: str
    def __init__(self, name: _Optional[str] = ..., experiment_ids: _Optional[_Iterable[str]] = ..., source_type: _Optional[_Union[_datasets_pb2.DatasetRecordSource.SourceType, str]] = ..., source: _Optional[str] = ..., schema: _Optional[str] = ..., profile: _Optional[str] = ..., created_by: _Optional[str] = ..., tags: _Optional[str] = ...) -> None: ...

class GetDataset(_message.Message):
    __slots__ = ("dataset_id", "page_token")
    class Response(_message.Message):
        __slots__ = ("dataset", "next_page_token")
        DATASET_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        dataset: _datasets_pb2.Dataset
        next_page_token: str
        def __init__(self, dataset: _Optional[_Union[_datasets_pb2.Dataset, _Mapping]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    dataset_id: str
    page_token: str
    def __init__(self, dataset_id: _Optional[str] = ..., page_token: _Optional[str] = ...) -> None: ...

class DeleteDataset(_message.Message):
    __slots__ = ("dataset_id",)
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    dataset_id: str
    def __init__(self, dataset_id: _Optional[str] = ...) -> None: ...

class SearchEvaluationDatasets(_message.Message):
    __slots__ = ("experiment_ids", "filter_string", "max_results", "order_by", "page_token")
    class Response(_message.Message):
        __slots__ = ("datasets", "next_page_token")
        DATASETS_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        datasets: _containers.RepeatedCompositeFieldContainer[_datasets_pb2.Dataset]
        next_page_token: str
        def __init__(self, datasets: _Optional[_Iterable[_Union[_datasets_pb2.Dataset, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    EXPERIMENT_IDS_FIELD_NUMBER: _ClassVar[int]
    FILTER_STRING_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    ORDER_BY_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    experiment_ids: _containers.RepeatedScalarFieldContainer[str]
    filter_string: str
    max_results: int
    order_by: _containers.RepeatedScalarFieldContainer[str]
    page_token: str
    def __init__(self, experiment_ids: _Optional[_Iterable[str]] = ..., filter_string: _Optional[str] = ..., max_results: _Optional[int] = ..., order_by: _Optional[_Iterable[str]] = ..., page_token: _Optional[str] = ...) -> None: ...

class SetDatasetTags(_message.Message):
    __slots__ = ("dataset_id", "tags")
    class Response(_message.Message):
        __slots__ = ("dataset",)
        DATASET_FIELD_NUMBER: _ClassVar[int]
        dataset: _datasets_pb2.Dataset
        def __init__(self, dataset: _Optional[_Union[_datasets_pb2.Dataset, _Mapping]] = ...) -> None: ...
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    dataset_id: str
    tags: str
    def __init__(self, dataset_id: _Optional[str] = ..., tags: _Optional[str] = ...) -> None: ...

class DeleteDatasetTag(_message.Message):
    __slots__ = ("dataset_id", "key")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    dataset_id: str
    key: str
    def __init__(self, dataset_id: _Optional[str] = ..., key: _Optional[str] = ...) -> None: ...

class UpsertDatasetRecords(_message.Message):
    __slots__ = ("dataset_id", "records", "updated_by")
    class Response(_message.Message):
        __slots__ = ("inserted_count", "updated_count")
        INSERTED_COUNT_FIELD_NUMBER: _ClassVar[int]
        UPDATED_COUNT_FIELD_NUMBER: _ClassVar[int]
        inserted_count: int
        updated_count: int
        def __init__(self, inserted_count: _Optional[int] = ..., updated_count: _Optional[int] = ...) -> None: ...
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    RECORDS_FIELD_NUMBER: _ClassVar[int]
    UPDATED_BY_FIELD_NUMBER: _ClassVar[int]
    dataset_id: str
    records: str
    updated_by: str
    def __init__(self, dataset_id: _Optional[str] = ..., records: _Optional[str] = ..., updated_by: _Optional[str] = ...) -> None: ...

class GetDatasetExperimentIds(_message.Message):
    __slots__ = ("dataset_id",)
    class Response(_message.Message):
        __slots__ = ("experiment_ids",)
        EXPERIMENT_IDS_FIELD_NUMBER: _ClassVar[int]
        experiment_ids: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, experiment_ids: _Optional[_Iterable[str]] = ...) -> None: ...
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    dataset_id: str
    def __init__(self, dataset_id: _Optional[str] = ...) -> None: ...

class GetDatasetRecords(_message.Message):
    __slots__ = ("dataset_id", "max_results", "page_token")
    class Response(_message.Message):
        __slots__ = ("records", "next_page_token")
        RECORDS_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        records: str
        next_page_token: str
        def __init__(self, records: _Optional[str] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    dataset_id: str
    max_results: int
    page_token: str
    def __init__(self, dataset_id: _Optional[str] = ..., max_results: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...

class AddDatasetToExperiments(_message.Message):
    __slots__ = ("dataset_id", "experiment_ids")
    class Response(_message.Message):
        __slots__ = ("dataset",)
        DATASET_FIELD_NUMBER: _ClassVar[int]
        dataset: Dataset
        def __init__(self, dataset: _Optional[_Union[Dataset, _Mapping]] = ...) -> None: ...
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_IDS_FIELD_NUMBER: _ClassVar[int]
    dataset_id: str
    experiment_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, dataset_id: _Optional[str] = ..., experiment_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class RemoveDatasetFromExperiments(_message.Message):
    __slots__ = ("dataset_id", "experiment_ids")
    class Response(_message.Message):
        __slots__ = ("dataset",)
        DATASET_FIELD_NUMBER: _ClassVar[int]
        dataset: Dataset
        def __init__(self, dataset: _Optional[_Union[Dataset, _Mapping]] = ...) -> None: ...
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_IDS_FIELD_NUMBER: _ClassVar[int]
    dataset_id: str
    experiment_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, dataset_id: _Optional[str] = ..., experiment_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class RegisterScorer(_message.Message):
    __slots__ = ("experiment_id", "name", "serialized_scorer")
    class Response(_message.Message):
        __slots__ = ("version",)
        VERSION_FIELD_NUMBER: _ClassVar[int]
        version: int
        def __init__(self, version: _Optional[int] = ...) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    SERIALIZED_SCORER_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    name: str
    serialized_scorer: str
    def __init__(self, experiment_id: _Optional[str] = ..., name: _Optional[str] = ..., serialized_scorer: _Optional[str] = ...) -> None: ...

class ListScorers(_message.Message):
    __slots__ = ("experiment_id",)
    class Response(_message.Message):
        __slots__ = ("scorers",)
        SCORERS_FIELD_NUMBER: _ClassVar[int]
        scorers: _containers.RepeatedCompositeFieldContainer[Scorer]
        def __init__(self, scorers: _Optional[_Iterable[_Union[Scorer, _Mapping]]] = ...) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    def __init__(self, experiment_id: _Optional[str] = ...) -> None: ...

class ListScorerVersions(_message.Message):
    __slots__ = ("experiment_id", "name")
    class Response(_message.Message):
        __slots__ = ("scorers",)
        SCORERS_FIELD_NUMBER: _ClassVar[int]
        scorers: _containers.RepeatedCompositeFieldContainer[Scorer]
        def __init__(self, scorers: _Optional[_Iterable[_Union[Scorer, _Mapping]]] = ...) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    name: str
    def __init__(self, experiment_id: _Optional[str] = ..., name: _Optional[str] = ...) -> None: ...

class GetScorer(_message.Message):
    __slots__ = ("experiment_id", "name", "version")
    class Response(_message.Message):
        __slots__ = ("scorer",)
        SCORER_FIELD_NUMBER: _ClassVar[int]
        scorer: Scorer
        def __init__(self, scorer: _Optional[_Union[Scorer, _Mapping]] = ...) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    name: str
    version: int
    def __init__(self, experiment_id: _Optional[str] = ..., name: _Optional[str] = ..., version: _Optional[int] = ...) -> None: ...

class DeleteScorer(_message.Message):
    __slots__ = ("experiment_id", "name", "version")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    name: str
    version: int
    def __init__(self, experiment_id: _Optional[str] = ..., name: _Optional[str] = ..., version: _Optional[int] = ...) -> None: ...

class Scorer(_message.Message):
    __slots__ = ("experiment_id", "scorer_name", "scorer_version", "serialized_scorer", "creation_time")
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    SCORER_NAME_FIELD_NUMBER: _ClassVar[int]
    SCORER_VERSION_FIELD_NUMBER: _ClassVar[int]
    SERIALIZED_SCORER_FIELD_NUMBER: _ClassVar[int]
    CREATION_TIME_FIELD_NUMBER: _ClassVar[int]
    experiment_id: int
    scorer_name: str
    scorer_version: int
    serialized_scorer: str
    creation_time: int
    def __init__(self, experiment_id: _Optional[int] = ..., scorer_name: _Optional[str] = ..., scorer_version: _Optional[int] = ..., serialized_scorer: _Optional[str] = ..., creation_time: _Optional[int] = ...) -> None: ...

class MlflowService(_service.service): ...

class MlflowService_Stub(MlflowService): ...

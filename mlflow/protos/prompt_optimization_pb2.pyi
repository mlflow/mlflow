from scalapb import scalapb_pb2 as _scalapb_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class OptimizationJobStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OPTIMIZATION_JOB_STATUS_UNSPECIFIED: _ClassVar[OptimizationJobStatus]
    OPTIMIZATION_JOB_STATUS_PENDING: _ClassVar[OptimizationJobStatus]
    OPTIMIZATION_JOB_STATUS_IN_PROGRESS: _ClassVar[OptimizationJobStatus]
    OPTIMIZATION_JOB_STATUS_COMPLETED: _ClassVar[OptimizationJobStatus]
    OPTIMIZATION_JOB_STATUS_FAILED: _ClassVar[OptimizationJobStatus]
    OPTIMIZATION_JOB_STATUS_CANCELED: _ClassVar[OptimizationJobStatus]

class OptimizerType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OPTIMIZER_TYPE_UNSPECIFIED: _ClassVar[OptimizerType]
    OPTIMIZER_TYPE_GEPA: _ClassVar[OptimizerType]
    OPTIMIZER_TYPE_METAPROMPT: _ClassVar[OptimizerType]
OPTIMIZATION_JOB_STATUS_UNSPECIFIED: OptimizationJobStatus
OPTIMIZATION_JOB_STATUS_PENDING: OptimizationJobStatus
OPTIMIZATION_JOB_STATUS_IN_PROGRESS: OptimizationJobStatus
OPTIMIZATION_JOB_STATUS_COMPLETED: OptimizationJobStatus
OPTIMIZATION_JOB_STATUS_FAILED: OptimizationJobStatus
OPTIMIZATION_JOB_STATUS_CANCELED: OptimizationJobStatus
OPTIMIZER_TYPE_UNSPECIFIED: OptimizerType
OPTIMIZER_TYPE_GEPA: OptimizerType
OPTIMIZER_TYPE_METAPROMPT: OptimizerType

class OptimizationJobTag(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class OptimizationJobConfig(_message.Message):
    __slots__ = ("target_prompt_uri", "optimizer_type", "optimizer_config_json")
    TARGET_PROMPT_URI_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZER_TYPE_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZER_CONFIG_JSON_FIELD_NUMBER: _ClassVar[int]
    target_prompt_uri: str
    optimizer_type: OptimizerType
    optimizer_config_json: str
    def __init__(self, target_prompt_uri: _Optional[str] = ..., optimizer_type: _Optional[_Union[OptimizerType, str]] = ..., optimizer_config_json: _Optional[str] = ...) -> None: ...

class OptimizationJob(_message.Message):
    __slots__ = ("job_id", "run_id", "status", "experiment_id", "source_prompt_uri", "optimized_prompt_uri", "config", "creation_timestamp_ms", "completion_timestamp_ms", "error_message", "tags", "initial_eval_score", "final_eval_score")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_PROMPT_URI_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZED_PROMPT_URI_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    CREATION_TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    INITIAL_EVAL_SCORE_FIELD_NUMBER: _ClassVar[int]
    FINAL_EVAL_SCORE_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    run_id: str
    status: OptimizationJobStatus
    experiment_id: str
    source_prompt_uri: str
    optimized_prompt_uri: str
    config: OptimizationJobConfig
    creation_timestamp_ms: int
    completion_timestamp_ms: int
    error_message: str
    tags: _containers.RepeatedCompositeFieldContainer[OptimizationJobTag]
    initial_eval_score: float
    final_eval_score: float
    def __init__(self, job_id: _Optional[str] = ..., run_id: _Optional[str] = ..., status: _Optional[_Union[OptimizationJobStatus, str]] = ..., experiment_id: _Optional[str] = ..., source_prompt_uri: _Optional[str] = ..., optimized_prompt_uri: _Optional[str] = ..., config: _Optional[_Union[OptimizationJobConfig, _Mapping]] = ..., creation_timestamp_ms: _Optional[int] = ..., completion_timestamp_ms: _Optional[int] = ..., error_message: _Optional[str] = ..., tags: _Optional[_Iterable[_Union[OptimizationJobTag, _Mapping]]] = ..., initial_eval_score: _Optional[float] = ..., final_eval_score: _Optional[float] = ...) -> None: ...

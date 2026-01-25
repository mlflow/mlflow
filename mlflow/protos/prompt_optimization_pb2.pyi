import jobs_pb2 as _jobs_pb2
from scalapb import scalapb_pb2 as _scalapb_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class OptimizerType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OPTIMIZER_TYPE_UNSPECIFIED: _ClassVar[OptimizerType]
    OPTIMIZER_TYPE_GEPA: _ClassVar[OptimizerType]
    OPTIMIZER_TYPE_METAPROMPT: _ClassVar[OptimizerType]
OPTIMIZER_TYPE_UNSPECIFIED: OptimizerType
OPTIMIZER_TYPE_GEPA: OptimizerType
OPTIMIZER_TYPE_METAPROMPT: OptimizerType

class PromptOptimizationJobTag(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class PromptOptimizationJobConfig(_message.Message):
    __slots__ = ("optimizer_type", "dataset_id", "scorers", "optimizer_config_json")
    OPTIMIZER_TYPE_FIELD_NUMBER: _ClassVar[int]
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    SCORERS_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZER_CONFIG_JSON_FIELD_NUMBER: _ClassVar[int]
    optimizer_type: OptimizerType
    dataset_id: str
    scorers: _containers.RepeatedScalarFieldContainer[str]
    optimizer_config_json: str
    def __init__(self, optimizer_type: _Optional[_Union[OptimizerType, str]] = ..., dataset_id: _Optional[str] = ..., scorers: _Optional[_Iterable[str]] = ..., optimizer_config_json: _Optional[str] = ...) -> None: ...

class PromptOptimizationJob(_message.Message):
    __slots__ = ("job_id", "run_id", "state", "experiment_id", "source_prompt_uri", "optimized_prompt_uri", "config", "creation_timestamp_ms", "completion_timestamp_ms", "tags", "initial_eval_scores", "final_eval_scores")
    class InitialEvalScoresEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    class FinalEvalScoresEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_PROMPT_URI_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZED_PROMPT_URI_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    CREATION_TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    INITIAL_EVAL_SCORES_FIELD_NUMBER: _ClassVar[int]
    FINAL_EVAL_SCORES_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    run_id: str
    state: _jobs_pb2.JobState
    experiment_id: str
    source_prompt_uri: str
    optimized_prompt_uri: str
    config: PromptOptimizationJobConfig
    creation_timestamp_ms: int
    completion_timestamp_ms: int
    tags: _containers.RepeatedCompositeFieldContainer[PromptOptimizationJobTag]
    initial_eval_scores: _containers.ScalarMap[str, float]
    final_eval_scores: _containers.ScalarMap[str, float]
    def __init__(self, job_id: _Optional[str] = ..., run_id: _Optional[str] = ..., state: _Optional[_Union[_jobs_pb2.JobState, _Mapping]] = ..., experiment_id: _Optional[str] = ..., source_prompt_uri: _Optional[str] = ..., optimized_prompt_uri: _Optional[str] = ..., config: _Optional[_Union[PromptOptimizationJobConfig, _Mapping]] = ..., creation_timestamp_ms: _Optional[int] = ..., completion_timestamp_ms: _Optional[int] = ..., tags: _Optional[_Iterable[_Union[PromptOptimizationJobTag, _Mapping]]] = ..., initial_eval_scores: _Optional[_Mapping[str, float]] = ..., final_eval_scores: _Optional[_Mapping[str, float]] = ...) -> None: ...

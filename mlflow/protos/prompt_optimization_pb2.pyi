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
OPTIMIZATION_JOB_STATUS_UNSPECIFIED: OptimizationJobStatus
OPTIMIZATION_JOB_STATUS_PENDING: OptimizationJobStatus
OPTIMIZATION_JOB_STATUS_IN_PROGRESS: OptimizationJobStatus
OPTIMIZATION_JOB_STATUS_COMPLETED: OptimizationJobStatus
OPTIMIZATION_JOB_STATUS_FAILED: OptimizationJobStatus
OPTIMIZATION_JOB_STATUS_CANCELED: OptimizationJobStatus
OPTIMIZER_TYPE_UNSPECIFIED: OptimizerType
OPTIMIZER_TYPE_GEPA: OptimizerType

class OptimizationJobTag(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class PromptModelConfig(_message.Message):
    __slots__ = ("provider", "model_name", "temperature", "max_tokens", "top_p", "top_k", "frequency_penalty", "presence_penalty", "stop_sequences", "extra_params_json")
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    FREQUENCY_PENALTY_FIELD_NUMBER: _ClassVar[int]
    PRESENCE_PENALTY_FIELD_NUMBER: _ClassVar[int]
    STOP_SEQUENCES_FIELD_NUMBER: _ClassVar[int]
    EXTRA_PARAMS_JSON_FIELD_NUMBER: _ClassVar[int]
    provider: str
    model_name: str
    temperature: float
    max_tokens: int
    top_p: float
    top_k: int
    frequency_penalty: float
    presence_penalty: float
    stop_sequences: _containers.RepeatedScalarFieldContainer[str]
    extra_params_json: str
    def __init__(self, provider: _Optional[str] = ..., model_name: _Optional[str] = ..., temperature: _Optional[float] = ..., max_tokens: _Optional[int] = ..., top_p: _Optional[float] = ..., top_k: _Optional[int] = ..., frequency_penalty: _Optional[float] = ..., presence_penalty: _Optional[float] = ..., stop_sequences: _Optional[_Iterable[str]] = ..., extra_params_json: _Optional[str] = ...) -> None: ...

class PromptVersionTag(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class PromptVersion(_message.Message):
    __slots__ = ("name", "version", "template", "prompt_type", "model_config", "response_format_json", "commit_message", "creation_timestamp_ms", "last_updated_timestamp_ms", "user_id", "tags", "aliases")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TYPE_FIELD_NUMBER: _ClassVar[int]
    MODEL_CONFIG_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FORMAT_JSON_FIELD_NUMBER: _ClassVar[int]
    COMMIT_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CREATION_TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    LAST_UPDATED_TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    ALIASES_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: int
    template: str
    prompt_type: str
    model_config: PromptModelConfig
    response_format_json: str
    commit_message: str
    creation_timestamp_ms: int
    last_updated_timestamp_ms: int
    user_id: str
    tags: _containers.RepeatedCompositeFieldContainer[PromptVersionTag]
    aliases: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, name: _Optional[str] = ..., version: _Optional[int] = ..., template: _Optional[str] = ..., prompt_type: _Optional[str] = ..., model_config: _Optional[_Union[PromptModelConfig, _Mapping]] = ..., response_format_json: _Optional[str] = ..., commit_message: _Optional[str] = ..., creation_timestamp_ms: _Optional[int] = ..., last_updated_timestamp_ms: _Optional[int] = ..., user_id: _Optional[str] = ..., tags: _Optional[_Iterable[_Union[PromptVersionTag, _Mapping]]] = ..., aliases: _Optional[_Iterable[str]] = ...) -> None: ...

class OptimizationJobConfig(_message.Message):
    __slots__ = ("target_prompt", "optimizer_type", "optimizer_config_json")
    TARGET_PROMPT_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZER_TYPE_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZER_CONFIG_JSON_FIELD_NUMBER: _ClassVar[int]
    target_prompt: PromptVersion
    optimizer_type: OptimizerType
    optimizer_config_json: str
    def __init__(self, target_prompt: _Optional[_Union[PromptVersion, _Mapping]] = ..., optimizer_type: _Optional[_Union[OptimizerType, str]] = ..., optimizer_config_json: _Optional[str] = ...) -> None: ...

class OptimizationJob(_message.Message):
    __slots__ = ("job_id", "status", "creation_timestamp_ms", "completion_timestamp_ms", "experiment_id", "run_id", "config", "tags", "source_prompt", "optimized_prompt", "error_message")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CREATION_TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_PROMPT_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZED_PROMPT_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    status: OptimizationJobStatus
    creation_timestamp_ms: int
    completion_timestamp_ms: int
    experiment_id: str
    run_id: str
    config: OptimizationJobConfig
    tags: _containers.RepeatedCompositeFieldContainer[OptimizationJobTag]
    source_prompt: PromptVersion
    optimized_prompt: PromptVersion
    error_message: str
    def __init__(self, job_id: _Optional[str] = ..., status: _Optional[_Union[OptimizationJobStatus, str]] = ..., creation_timestamp_ms: _Optional[int] = ..., completion_timestamp_ms: _Optional[int] = ..., experiment_id: _Optional[str] = ..., run_id: _Optional[str] = ..., config: _Optional[_Union[OptimizationJobConfig, _Mapping]] = ..., tags: _Optional[_Iterable[_Union[OptimizationJobTag, _Mapping]]] = ..., source_prompt: _Optional[_Union[PromptVersion, _Mapping]] = ..., optimized_prompt: _Optional[_Union[PromptVersion, _Mapping]] = ..., error_message: _Optional[str] = ...) -> None: ...

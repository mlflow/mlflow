import databricks_pb2 as _databricks_pb2
from scalapb import scalapb_pb2 as _scalapb_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import service as _service
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ModelVersionStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PENDING_REGISTRATION: _ClassVar[ModelVersionStatus]
    FAILED_REGISTRATION: _ClassVar[ModelVersionStatus]
    READY: _ClassVar[ModelVersionStatus]
PENDING_REGISTRATION: ModelVersionStatus
FAILED_REGISTRATION: ModelVersionStatus
READY: ModelVersionStatus

class RegisteredModel(_message.Message):
    __slots__ = ("name", "creation_timestamp", "last_updated_timestamp", "user_id", "description", "latest_versions", "tags", "aliases", "deployment_job_id", "deployment_job_state")
    NAME_FIELD_NUMBER: _ClassVar[int]
    CREATION_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    LAST_UPDATED_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    LATEST_VERSIONS_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    ALIASES_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_JOB_STATE_FIELD_NUMBER: _ClassVar[int]
    name: str
    creation_timestamp: int
    last_updated_timestamp: int
    user_id: str
    description: str
    latest_versions: _containers.RepeatedCompositeFieldContainer[ModelVersion]
    tags: _containers.RepeatedCompositeFieldContainer[RegisteredModelTag]
    aliases: _containers.RepeatedCompositeFieldContainer[RegisteredModelAlias]
    deployment_job_id: str
    deployment_job_state: DeploymentJobConnection.State
    def __init__(self, name: _Optional[str] = ..., creation_timestamp: _Optional[int] = ..., last_updated_timestamp: _Optional[int] = ..., user_id: _Optional[str] = ..., description: _Optional[str] = ..., latest_versions: _Optional[_Iterable[_Union[ModelVersion, _Mapping]]] = ..., tags: _Optional[_Iterable[_Union[RegisteredModelTag, _Mapping]]] = ..., aliases: _Optional[_Iterable[_Union[RegisteredModelAlias, _Mapping]]] = ..., deployment_job_id: _Optional[str] = ..., deployment_job_state: _Optional[_Union[DeploymentJobConnection.State, str]] = ...) -> None: ...

class ModelVersion(_message.Message):
    __slots__ = ("name", "version", "creation_timestamp", "last_updated_timestamp", "user_id", "current_stage", "description", "source", "run_id", "status", "status_message", "tags", "run_link", "aliases", "model_id", "model_params", "model_metrics", "deployment_job_state")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    CREATION_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    LAST_UPDATED_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    CURRENT_STAGE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    STATUS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    RUN_LINK_FIELD_NUMBER: _ClassVar[int]
    ALIASES_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_PARAMS_FIELD_NUMBER: _ClassVar[int]
    MODEL_METRICS_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_JOB_STATE_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    creation_timestamp: int
    last_updated_timestamp: int
    user_id: str
    current_stage: str
    description: str
    source: str
    run_id: str
    status: ModelVersionStatus
    status_message: str
    tags: _containers.RepeatedCompositeFieldContainer[ModelVersionTag]
    run_link: str
    aliases: _containers.RepeatedScalarFieldContainer[str]
    model_id: str
    model_params: _containers.RepeatedCompositeFieldContainer[ModelParam]
    model_metrics: _containers.RepeatedCompositeFieldContainer[ModelMetric]
    deployment_job_state: ModelVersionDeploymentJobState
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., creation_timestamp: _Optional[int] = ..., last_updated_timestamp: _Optional[int] = ..., user_id: _Optional[str] = ..., current_stage: _Optional[str] = ..., description: _Optional[str] = ..., source: _Optional[str] = ..., run_id: _Optional[str] = ..., status: _Optional[_Union[ModelVersionStatus, str]] = ..., status_message: _Optional[str] = ..., tags: _Optional[_Iterable[_Union[ModelVersionTag, _Mapping]]] = ..., run_link: _Optional[str] = ..., aliases: _Optional[_Iterable[str]] = ..., model_id: _Optional[str] = ..., model_params: _Optional[_Iterable[_Union[ModelParam, _Mapping]]] = ..., model_metrics: _Optional[_Iterable[_Union[ModelMetric, _Mapping]]] = ..., deployment_job_state: _Optional[_Union[ModelVersionDeploymentJobState, _Mapping]] = ...) -> None: ...

class DeploymentJobConnection(_message.Message):
    __slots__ = ()
    class State(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        DEPLOYMENT_JOB_CONNECTION_STATE_UNSPECIFIED: _ClassVar[DeploymentJobConnection.State]
        NOT_SET_UP: _ClassVar[DeploymentJobConnection.State]
        CONNECTED: _ClassVar[DeploymentJobConnection.State]
        NOT_FOUND: _ClassVar[DeploymentJobConnection.State]
        REQUIRED_PARAMETERS_CHANGED: _ClassVar[DeploymentJobConnection.State]
    DEPLOYMENT_JOB_CONNECTION_STATE_UNSPECIFIED: DeploymentJobConnection.State
    NOT_SET_UP: DeploymentJobConnection.State
    CONNECTED: DeploymentJobConnection.State
    NOT_FOUND: DeploymentJobConnection.State
    REQUIRED_PARAMETERS_CHANGED: DeploymentJobConnection.State
    def __init__(self) -> None: ...

class ModelVersionDeploymentJobState(_message.Message):
    __slots__ = ("job_id", "run_id", "job_state", "run_state", "current_task_name")
    class DeploymentJobRunState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        DEPLOYMENT_JOB_RUN_STATE_UNSPECIFIED: _ClassVar[ModelVersionDeploymentJobState.DeploymentJobRunState]
        NO_VALID_DEPLOYMENT_JOB_FOUND: _ClassVar[ModelVersionDeploymentJobState.DeploymentJobRunState]
        RUNNING: _ClassVar[ModelVersionDeploymentJobState.DeploymentJobRunState]
        SUCCEEDED: _ClassVar[ModelVersionDeploymentJobState.DeploymentJobRunState]
        FAILED: _ClassVar[ModelVersionDeploymentJobState.DeploymentJobRunState]
        PENDING: _ClassVar[ModelVersionDeploymentJobState.DeploymentJobRunState]
        APPROVAL: _ClassVar[ModelVersionDeploymentJobState.DeploymentJobRunState]
    DEPLOYMENT_JOB_RUN_STATE_UNSPECIFIED: ModelVersionDeploymentJobState.DeploymentJobRunState
    NO_VALID_DEPLOYMENT_JOB_FOUND: ModelVersionDeploymentJobState.DeploymentJobRunState
    RUNNING: ModelVersionDeploymentJobState.DeploymentJobRunState
    SUCCEEDED: ModelVersionDeploymentJobState.DeploymentJobRunState
    FAILED: ModelVersionDeploymentJobState.DeploymentJobRunState
    PENDING: ModelVersionDeploymentJobState.DeploymentJobRunState
    APPROVAL: ModelVersionDeploymentJobState.DeploymentJobRunState
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_STATE_FIELD_NUMBER: _ClassVar[int]
    RUN_STATE_FIELD_NUMBER: _ClassVar[int]
    CURRENT_TASK_NAME_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    run_id: str
    job_state: DeploymentJobConnection.State
    run_state: ModelVersionDeploymentJobState.DeploymentJobRunState
    current_task_name: str
    def __init__(self, job_id: _Optional[str] = ..., run_id: _Optional[str] = ..., job_state: _Optional[_Union[DeploymentJobConnection.State, str]] = ..., run_state: _Optional[_Union[ModelVersionDeploymentJobState.DeploymentJobRunState, str]] = ..., current_task_name: _Optional[str] = ...) -> None: ...

class CreateRegisteredModel(_message.Message):
    __slots__ = ("name", "tags", "description", "deployment_job_id")
    class Response(_message.Message):
        __slots__ = ("registered_model",)
        REGISTERED_MODEL_FIELD_NUMBER: _ClassVar[int]
        registered_model: RegisteredModel
        def __init__(self, registered_model: _Optional[_Union[RegisteredModel, _Mapping]] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    name: str
    tags: _containers.RepeatedCompositeFieldContainer[RegisteredModelTag]
    description: str
    deployment_job_id: str
    def __init__(self, name: _Optional[str] = ..., tags: _Optional[_Iterable[_Union[RegisteredModelTag, _Mapping]]] = ..., description: _Optional[str] = ..., deployment_job_id: _Optional[str] = ...) -> None: ...

class RenameRegisteredModel(_message.Message):
    __slots__ = ("name", "new_name")
    class Response(_message.Message):
        __slots__ = ("registered_model",)
        REGISTERED_MODEL_FIELD_NUMBER: _ClassVar[int]
        registered_model: RegisteredModel
        def __init__(self, registered_model: _Optional[_Union[RegisteredModel, _Mapping]] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    NEW_NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    new_name: str
    def __init__(self, name: _Optional[str] = ..., new_name: _Optional[str] = ...) -> None: ...

class UpdateRegisteredModel(_message.Message):
    __slots__ = ("name", "description", "deployment_job_id")
    class Response(_message.Message):
        __slots__ = ("registered_model",)
        REGISTERED_MODEL_FIELD_NUMBER: _ClassVar[int]
        registered_model: RegisteredModel
        def __init__(self, registered_model: _Optional[_Union[RegisteredModel, _Mapping]] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    name: str
    description: str
    deployment_job_id: str
    def __init__(self, name: _Optional[str] = ..., description: _Optional[str] = ..., deployment_job_id: _Optional[str] = ...) -> None: ...

class DeleteRegisteredModel(_message.Message):
    __slots__ = ("name",)
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class GetRegisteredModel(_message.Message):
    __slots__ = ("name",)
    class Response(_message.Message):
        __slots__ = ("registered_model",)
        REGISTERED_MODEL_FIELD_NUMBER: _ClassVar[int]
        registered_model: RegisteredModel
        def __init__(self, registered_model: _Optional[_Union[RegisteredModel, _Mapping]] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class SearchRegisteredModels(_message.Message):
    __slots__ = ("filter", "max_results", "order_by", "page_token")
    class Response(_message.Message):
        __slots__ = ("registered_models", "next_page_token")
        REGISTERED_MODELS_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        registered_models: _containers.RepeatedCompositeFieldContainer[RegisteredModel]
        next_page_token: str
        def __init__(self, registered_models: _Optional[_Iterable[_Union[RegisteredModel, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    FILTER_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    ORDER_BY_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    filter: str
    max_results: int
    order_by: _containers.RepeatedScalarFieldContainer[str]
    page_token: str
    def __init__(self, filter: _Optional[str] = ..., max_results: _Optional[int] = ..., order_by: _Optional[_Iterable[str]] = ..., page_token: _Optional[str] = ...) -> None: ...

class GetLatestVersions(_message.Message):
    __slots__ = ("name", "stages")
    class Response(_message.Message):
        __slots__ = ("model_versions",)
        MODEL_VERSIONS_FIELD_NUMBER: _ClassVar[int]
        model_versions: _containers.RepeatedCompositeFieldContainer[ModelVersion]
        def __init__(self, model_versions: _Optional[_Iterable[_Union[ModelVersion, _Mapping]]] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    STAGES_FIELD_NUMBER: _ClassVar[int]
    name: str
    stages: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, name: _Optional[str] = ..., stages: _Optional[_Iterable[str]] = ...) -> None: ...

class CreateModelVersion(_message.Message):
    __slots__ = ("name", "source", "run_id", "tags", "run_link", "description", "model_id")
    class Response(_message.Message):
        __slots__ = ("model_version",)
        MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
        model_version: ModelVersion
        def __init__(self, model_version: _Optional[_Union[ModelVersion, _Mapping]] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    RUN_LINK_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    name: str
    source: str
    run_id: str
    tags: _containers.RepeatedCompositeFieldContainer[ModelVersionTag]
    run_link: str
    description: str
    model_id: str
    def __init__(self, name: _Optional[str] = ..., source: _Optional[str] = ..., run_id: _Optional[str] = ..., tags: _Optional[_Iterable[_Union[ModelVersionTag, _Mapping]]] = ..., run_link: _Optional[str] = ..., description: _Optional[str] = ..., model_id: _Optional[str] = ...) -> None: ...

class UpdateModelVersion(_message.Message):
    __slots__ = ("name", "version", "description")
    class Response(_message.Message):
        __slots__ = ("model_version",)
        MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
        model_version: ModelVersion
        def __init__(self, model_version: _Optional[_Union[ModelVersion, _Mapping]] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    description: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., description: _Optional[str] = ...) -> None: ...

class TransitionModelVersionStage(_message.Message):
    __slots__ = ("name", "version", "stage", "archive_existing_versions")
    class Response(_message.Message):
        __slots__ = ("model_version",)
        MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
        model_version: ModelVersion
        def __init__(self, model_version: _Optional[_Union[ModelVersion, _Mapping]] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    STAGE_FIELD_NUMBER: _ClassVar[int]
    ARCHIVE_EXISTING_VERSIONS_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    stage: str
    archive_existing_versions: bool
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., stage: _Optional[str] = ..., archive_existing_versions: bool = ...) -> None: ...

class DeleteModelVersion(_message.Message):
    __slots__ = ("name", "version")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class GetModelVersion(_message.Message):
    __slots__ = ("name", "version")
    class Response(_message.Message):
        __slots__ = ("model_version",)
        MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
        model_version: ModelVersion
        def __init__(self, model_version: _Optional[_Union[ModelVersion, _Mapping]] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class SearchModelVersions(_message.Message):
    __slots__ = ("filter", "max_results", "order_by", "page_token")
    class Response(_message.Message):
        __slots__ = ("model_versions", "next_page_token")
        MODEL_VERSIONS_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        model_versions: _containers.RepeatedCompositeFieldContainer[ModelVersion]
        next_page_token: str
        def __init__(self, model_versions: _Optional[_Iterable[_Union[ModelVersion, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    FILTER_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    ORDER_BY_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    filter: str
    max_results: int
    order_by: _containers.RepeatedScalarFieldContainer[str]
    page_token: str
    def __init__(self, filter: _Optional[str] = ..., max_results: _Optional[int] = ..., order_by: _Optional[_Iterable[str]] = ..., page_token: _Optional[str] = ...) -> None: ...

class GetModelVersionDownloadUri(_message.Message):
    __slots__ = ("name", "version")
    class Response(_message.Message):
        __slots__ = ("artifact_uri",)
        ARTIFACT_URI_FIELD_NUMBER: _ClassVar[int]
        artifact_uri: str
        def __init__(self, artifact_uri: _Optional[str] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class ModelVersionTag(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class ModelParam(_message.Message):
    __slots__ = ("name", "value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: str
    def __init__(self, name: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class ModelMetric(_message.Message):
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

class RegisteredModelTag(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class SetRegisteredModelTag(_message.Message):
    __slots__ = ("name", "key", "value")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    key: str
    value: str
    def __init__(self, name: _Optional[str] = ..., key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class SetModelVersionTag(_message.Message):
    __slots__ = ("name", "version", "key", "value")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    key: str
    value: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class DeleteRegisteredModelTag(_message.Message):
    __slots__ = ("name", "key")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    name: str
    key: str
    def __init__(self, name: _Optional[str] = ..., key: _Optional[str] = ...) -> None: ...

class DeleteModelVersionTag(_message.Message):
    __slots__ = ("name", "version", "key")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    key: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., key: _Optional[str] = ...) -> None: ...

class RegisteredModelAlias(_message.Message):
    __slots__ = ("alias", "version")
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    alias: str
    version: str
    def __init__(self, alias: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class SetRegisteredModelAlias(_message.Message):
    __slots__ = ("name", "alias", "version")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    name: str
    alias: str
    version: str
    def __init__(self, name: _Optional[str] = ..., alias: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class DeleteRegisteredModelAlias(_message.Message):
    __slots__ = ("name", "alias")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    name: str
    alias: str
    def __init__(self, name: _Optional[str] = ..., alias: _Optional[str] = ...) -> None: ...

class GetModelVersionByAlias(_message.Message):
    __slots__ = ("name", "alias")
    class Response(_message.Message):
        __slots__ = ("model_version",)
        MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
        model_version: ModelVersion
        def __init__(self, model_version: _Optional[_Union[ModelVersion, _Mapping]] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    name: str
    alias: str
    def __init__(self, name: _Optional[str] = ..., alias: _Optional[str] = ...) -> None: ...

class ModelRegistryService(_service.service): ...

class ModelRegistryService_Stub(ModelRegistryService): ...

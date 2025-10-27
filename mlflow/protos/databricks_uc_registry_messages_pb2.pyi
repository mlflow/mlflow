from scalapb import scalapb_pb2 as _scalapb_pb2
import databricks_pb2 as _databricks_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ModelVersionStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNSPECIFIED: _ClassVar[ModelVersionStatus]
    PENDING_REGISTRATION: _ClassVar[ModelVersionStatus]
    FAILED_REGISTRATION: _ClassVar[ModelVersionStatus]
    READY: _ClassVar[ModelVersionStatus]

class ModelVersionOperation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MODEL_VERSION_OPERATION_UNSPECIFIED: _ClassVar[ModelVersionOperation]
    MODEL_VERSION_OPERATION_READ: _ClassVar[ModelVersionOperation]
    MODEL_VERSION_OPERATION_READ_WRITE: _ClassVar[ModelVersionOperation]

class StorageMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STORAGE_MODE_UNSPECIFIED: _ClassVar[StorageMode]
    CUSTOMER_HOSTED: _ClassVar[StorageMode]
    DEFAULT_STORAGE: _ClassVar[StorageMode]

class SseEncryptionAlgorithm(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SSE_ENCRYPTION_ALGORITHM_UNSPECIFIED: _ClassVar[SseEncryptionAlgorithm]
    AWS_SSE_S3: _ClassVar[SseEncryptionAlgorithm]
    AWS_SSE_KMS: _ClassVar[SseEncryptionAlgorithm]

class DependencyType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DEPENDENCY_TYPE_UNSPECIFIED: _ClassVar[DependencyType]
    DATABRICKS_VECTOR_INDEX: _ClassVar[DependencyType]
    DATABRICKS_MODEL_ENDPOINT: _ClassVar[DependencyType]
    DATABRICKS_UC_FUNCTION: _ClassVar[DependencyType]
    DATABRICKS_UC_CONNECTION: _ClassVar[DependencyType]
    DATABRICKS_TABLE: _ClassVar[DependencyType]

class TableType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TABLE: _ClassVar[TableType]
    PERSISTED_VIEW: _ClassVar[TableType]
    TEMP_VIEW: _ClassVar[TableType]
    MATERIALIZED_VIEW: _ClassVar[TableType]
    STREAMING_LIVE_TABLE: _ClassVar[TableType]
    PATH: _ClassVar[TableType]

class ModelVersionLineageDirection(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UPSTREAM: _ClassVar[ModelVersionLineageDirection]
    DOWNSTREAM: _ClassVar[ModelVersionLineageDirection]
UNSPECIFIED: ModelVersionStatus
PENDING_REGISTRATION: ModelVersionStatus
FAILED_REGISTRATION: ModelVersionStatus
READY: ModelVersionStatus
MODEL_VERSION_OPERATION_UNSPECIFIED: ModelVersionOperation
MODEL_VERSION_OPERATION_READ: ModelVersionOperation
MODEL_VERSION_OPERATION_READ_WRITE: ModelVersionOperation
STORAGE_MODE_UNSPECIFIED: StorageMode
CUSTOMER_HOSTED: StorageMode
DEFAULT_STORAGE: StorageMode
SSE_ENCRYPTION_ALGORITHM_UNSPECIFIED: SseEncryptionAlgorithm
AWS_SSE_S3: SseEncryptionAlgorithm
AWS_SSE_KMS: SseEncryptionAlgorithm
DEPENDENCY_TYPE_UNSPECIFIED: DependencyType
DATABRICKS_VECTOR_INDEX: DependencyType
DATABRICKS_MODEL_ENDPOINT: DependencyType
DATABRICKS_UC_FUNCTION: DependencyType
DATABRICKS_UC_CONNECTION: DependencyType
DATABRICKS_TABLE: DependencyType
TABLE: TableType
PERSISTED_VIEW: TableType
TEMP_VIEW: TableType
MATERIALIZED_VIEW: TableType
STREAMING_LIVE_TABLE: TableType
PATH: TableType
UPSTREAM: ModelVersionLineageDirection
DOWNSTREAM: ModelVersionLineageDirection

class RegisteredModel(_message.Message):
    __slots__ = ("name", "creation_timestamp", "last_updated_timestamp", "user_id", "description", "aliases", "tags", "deployment_job_id", "deployment_job_state")
    NAME_FIELD_NUMBER: _ClassVar[int]
    CREATION_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    LAST_UPDATED_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    ALIASES_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_JOB_STATE_FIELD_NUMBER: _ClassVar[int]
    name: str
    creation_timestamp: int
    last_updated_timestamp: int
    user_id: str
    description: str
    aliases: _containers.RepeatedCompositeFieldContainer[RegisteredModelAlias]
    tags: _containers.RepeatedCompositeFieldContainer[RegisteredModelTag]
    deployment_job_id: str
    deployment_job_state: DeploymentJobConnection.State
    def __init__(self, name: _Optional[str] = ..., creation_timestamp: _Optional[int] = ..., last_updated_timestamp: _Optional[int] = ..., user_id: _Optional[str] = ..., description: _Optional[str] = ..., aliases: _Optional[_Iterable[_Union[RegisteredModelAlias, _Mapping]]] = ..., tags: _Optional[_Iterable[_Union[RegisteredModelTag, _Mapping]]] = ..., deployment_job_id: _Optional[str] = ..., deployment_job_state: _Optional[_Union[DeploymentJobConnection.State, str]] = ...) -> None: ...

class RegisteredModelAlias(_message.Message):
    __slots__ = ("alias", "version")
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    alias: str
    version: str
    def __init__(self, alias: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class RegisteredModelTag(_message.Message):
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

class ModelVersionTag(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class ModelVersion(_message.Message):
    __slots__ = ("name", "version", "creation_timestamp", "last_updated_timestamp", "user_id", "description", "source", "run_id", "run_experiment_id", "run_tracking_server_id", "status", "status_message", "storage_location", "aliases", "tags", "model_id", "model_params", "model_metrics", "deployment_job_state")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    CREATION_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    LAST_UPDATED_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_TRACKING_SERVER_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    STATUS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    STORAGE_LOCATION_FIELD_NUMBER: _ClassVar[int]
    ALIASES_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_PARAMS_FIELD_NUMBER: _ClassVar[int]
    MODEL_METRICS_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_JOB_STATE_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    creation_timestamp: int
    last_updated_timestamp: int
    user_id: str
    description: str
    source: str
    run_id: str
    run_experiment_id: str
    run_tracking_server_id: str
    status: ModelVersionStatus
    status_message: str
    storage_location: str
    aliases: _containers.RepeatedCompositeFieldContainer[RegisteredModelAlias]
    tags: _containers.RepeatedCompositeFieldContainer[ModelVersionTag]
    model_id: str
    model_params: _containers.RepeatedCompositeFieldContainer[ModelParam]
    model_metrics: _containers.RepeatedCompositeFieldContainer[ModelMetric]
    deployment_job_state: ModelVersionDeploymentJobState
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., creation_timestamp: _Optional[int] = ..., last_updated_timestamp: _Optional[int] = ..., user_id: _Optional[str] = ..., description: _Optional[str] = ..., source: _Optional[str] = ..., run_id: _Optional[str] = ..., run_experiment_id: _Optional[str] = ..., run_tracking_server_id: _Optional[str] = ..., status: _Optional[_Union[ModelVersionStatus, str]] = ..., status_message: _Optional[str] = ..., storage_location: _Optional[str] = ..., aliases: _Optional[_Iterable[_Union[RegisteredModelAlias, _Mapping]]] = ..., tags: _Optional[_Iterable[_Union[ModelVersionTag, _Mapping]]] = ..., model_id: _Optional[str] = ..., model_params: _Optional[_Iterable[_Union[ModelParam, _Mapping]]] = ..., model_metrics: _Optional[_Iterable[_Union[ModelMetric, _Mapping]]] = ..., deployment_job_state: _Optional[_Union[ModelVersionDeploymentJobState, _Mapping]] = ...) -> None: ...

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

class TemporaryCredentials(_message.Message):
    __slots__ = ("aws_temp_credentials", "azure_user_delegation_sas", "gcp_oauth_token", "r2_temp_credentials", "expiration_time", "storage_mode", "encryption_details")
    AWS_TEMP_CREDENTIALS_FIELD_NUMBER: _ClassVar[int]
    AZURE_USER_DELEGATION_SAS_FIELD_NUMBER: _ClassVar[int]
    GCP_OAUTH_TOKEN_FIELD_NUMBER: _ClassVar[int]
    R2_TEMP_CREDENTIALS_FIELD_NUMBER: _ClassVar[int]
    EXPIRATION_TIME_FIELD_NUMBER: _ClassVar[int]
    STORAGE_MODE_FIELD_NUMBER: _ClassVar[int]
    ENCRYPTION_DETAILS_FIELD_NUMBER: _ClassVar[int]
    aws_temp_credentials: AwsCredentials
    azure_user_delegation_sas: AzureUserDelegationSAS
    gcp_oauth_token: GcpOauthToken
    r2_temp_credentials: R2Credentials
    expiration_time: int
    storage_mode: StorageMode
    encryption_details: EncryptionDetails
    def __init__(self, aws_temp_credentials: _Optional[_Union[AwsCredentials, _Mapping]] = ..., azure_user_delegation_sas: _Optional[_Union[AzureUserDelegationSAS, _Mapping]] = ..., gcp_oauth_token: _Optional[_Union[GcpOauthToken, _Mapping]] = ..., r2_temp_credentials: _Optional[_Union[R2Credentials, _Mapping]] = ..., expiration_time: _Optional[int] = ..., storage_mode: _Optional[_Union[StorageMode, str]] = ..., encryption_details: _Optional[_Union[EncryptionDetails, _Mapping]] = ...) -> None: ...

class AwsCredentials(_message.Message):
    __slots__ = ("access_key_id", "secret_access_key", "session_token")
    ACCESS_KEY_ID_FIELD_NUMBER: _ClassVar[int]
    SECRET_ACCESS_KEY_FIELD_NUMBER: _ClassVar[int]
    SESSION_TOKEN_FIELD_NUMBER: _ClassVar[int]
    access_key_id: str
    secret_access_key: str
    session_token: str
    def __init__(self, access_key_id: _Optional[str] = ..., secret_access_key: _Optional[str] = ..., session_token: _Optional[str] = ...) -> None: ...

class AzureUserDelegationSAS(_message.Message):
    __slots__ = ("sas_token",)
    SAS_TOKEN_FIELD_NUMBER: _ClassVar[int]
    sas_token: str
    def __init__(self, sas_token: _Optional[str] = ...) -> None: ...

class GcpOauthToken(_message.Message):
    __slots__ = ("oauth_token",)
    OAUTH_TOKEN_FIELD_NUMBER: _ClassVar[int]
    oauth_token: str
    def __init__(self, oauth_token: _Optional[str] = ...) -> None: ...

class R2Credentials(_message.Message):
    __slots__ = ("access_key_id", "secret_access_key", "session_token")
    ACCESS_KEY_ID_FIELD_NUMBER: _ClassVar[int]
    SECRET_ACCESS_KEY_FIELD_NUMBER: _ClassVar[int]
    SESSION_TOKEN_FIELD_NUMBER: _ClassVar[int]
    access_key_id: str
    secret_access_key: str
    session_token: str
    def __init__(self, access_key_id: _Optional[str] = ..., secret_access_key: _Optional[str] = ..., session_token: _Optional[str] = ...) -> None: ...

class EncryptionDetails(_message.Message):
    __slots__ = ("sse_encryption_details",)
    SSE_ENCRYPTION_DETAILS_FIELD_NUMBER: _ClassVar[int]
    sse_encryption_details: SseEncryptionDetails
    def __init__(self, sse_encryption_details: _Optional[_Union[SseEncryptionDetails, _Mapping]] = ...) -> None: ...

class SseEncryptionDetails(_message.Message):
    __slots__ = ("algorithm", "aws_kms_key_arn")
    ALGORITHM_FIELD_NUMBER: _ClassVar[int]
    AWS_KMS_KEY_ARN_FIELD_NUMBER: _ClassVar[int]
    algorithm: SseEncryptionAlgorithm
    aws_kms_key_arn: str
    def __init__(self, algorithm: _Optional[_Union[SseEncryptionAlgorithm, str]] = ..., aws_kms_key_arn: _Optional[str] = ...) -> None: ...

class CreateRegisteredModelRequest(_message.Message):
    __slots__ = ("name", "tags", "description", "deployment_job_id")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    name: str
    tags: _containers.RepeatedCompositeFieldContainer[RegisteredModelTag]
    description: str
    deployment_job_id: str
    def __init__(self, name: _Optional[str] = ..., tags: _Optional[_Iterable[_Union[RegisteredModelTag, _Mapping]]] = ..., description: _Optional[str] = ..., deployment_job_id: _Optional[str] = ...) -> None: ...

class CreateRegisteredModelResponse(_message.Message):
    __slots__ = ("registered_model",)
    REGISTERED_MODEL_FIELD_NUMBER: _ClassVar[int]
    registered_model: RegisteredModel
    def __init__(self, registered_model: _Optional[_Union[RegisteredModel, _Mapping]] = ...) -> None: ...

class UpdateRegisteredModelRequest(_message.Message):
    __slots__ = ("name", "new_name", "description", "deployment_job_id")
    NAME_FIELD_NUMBER: _ClassVar[int]
    NEW_NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    name: str
    new_name: str
    description: str
    deployment_job_id: str
    def __init__(self, name: _Optional[str] = ..., new_name: _Optional[str] = ..., description: _Optional[str] = ..., deployment_job_id: _Optional[str] = ...) -> None: ...

class UpdateRegisteredModelResponse(_message.Message):
    __slots__ = ("registered_model",)
    REGISTERED_MODEL_FIELD_NUMBER: _ClassVar[int]
    registered_model: RegisteredModel
    def __init__(self, registered_model: _Optional[_Union[RegisteredModel, _Mapping]] = ...) -> None: ...

class DeleteRegisteredModelRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class DeleteRegisteredModelResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetRegisteredModelRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class GetRegisteredModelResponse(_message.Message):
    __slots__ = ("registered_model",)
    REGISTERED_MODEL_FIELD_NUMBER: _ClassVar[int]
    registered_model: RegisteredModel
    def __init__(self, registered_model: _Optional[_Union[RegisteredModel, _Mapping]] = ...) -> None: ...

class SearchRegisteredModelsRequest(_message.Message):
    __slots__ = ("max_results", "page_token")
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    max_results: int
    page_token: str
    def __init__(self, max_results: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...

class SearchRegisteredModelsResponse(_message.Message):
    __slots__ = ("registered_models", "next_page_token")
    REGISTERED_MODELS_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    registered_models: _containers.RepeatedCompositeFieldContainer[RegisteredModel]
    next_page_token: str
    def __init__(self, registered_models: _Optional[_Iterable[_Union[RegisteredModel, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...

class Dependency(_message.Message):
    __slots__ = ("type", "name")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    type: DependencyType
    name: str
    def __init__(self, type: _Optional[_Union[DependencyType, str]] = ..., name: _Optional[str] = ...) -> None: ...

class CreateModelVersionRequest(_message.Message):
    __slots__ = ("name", "source", "run_id", "description", "run_tracking_server_id", "feature_deps", "tags", "model_version_dependencies", "model_id")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    RUN_TRACKING_SERVER_ID_FIELD_NUMBER: _ClassVar[int]
    FEATURE_DEPS_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_DEPENDENCIES_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    name: str
    source: str
    run_id: str
    description: str
    run_tracking_server_id: str
    feature_deps: str
    tags: _containers.RepeatedCompositeFieldContainer[ModelVersionTag]
    model_version_dependencies: _containers.RepeatedCompositeFieldContainer[Dependency]
    model_id: str
    def __init__(self, name: _Optional[str] = ..., source: _Optional[str] = ..., run_id: _Optional[str] = ..., description: _Optional[str] = ..., run_tracking_server_id: _Optional[str] = ..., feature_deps: _Optional[str] = ..., tags: _Optional[_Iterable[_Union[ModelVersionTag, _Mapping]]] = ..., model_version_dependencies: _Optional[_Iterable[_Union[Dependency, _Mapping]]] = ..., model_id: _Optional[str] = ...) -> None: ...

class CreateModelVersionResponse(_message.Message):
    __slots__ = ("model_version",)
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    model_version: ModelVersion
    def __init__(self, model_version: _Optional[_Union[ModelVersion, _Mapping]] = ...) -> None: ...

class UpdateModelVersionRequest(_message.Message):
    __slots__ = ("name", "version", "description")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    description: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., description: _Optional[str] = ...) -> None: ...

class UpdateModelVersionResponse(_message.Message):
    __slots__ = ("model_version",)
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    model_version: ModelVersion
    def __init__(self, model_version: _Optional[_Union[ModelVersion, _Mapping]] = ...) -> None: ...

class DeleteModelVersionRequest(_message.Message):
    __slots__ = ("name", "version")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class DeleteModelVersionResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetModelVersionRequest(_message.Message):
    __slots__ = ("name", "version")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class GetModelVersionResponse(_message.Message):
    __slots__ = ("model_version",)
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    model_version: ModelVersion
    def __init__(self, model_version: _Optional[_Union[ModelVersion, _Mapping]] = ...) -> None: ...

class SearchModelVersionsRequest(_message.Message):
    __slots__ = ("filter", "max_results", "page_token")
    FILTER_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    filter: str
    max_results: int
    page_token: str
    def __init__(self, filter: _Optional[str] = ..., max_results: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...

class SearchModelVersionsResponse(_message.Message):
    __slots__ = ("model_versions", "next_page_token")
    MODEL_VERSIONS_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    model_versions: _containers.RepeatedCompositeFieldContainer[ModelVersion]
    next_page_token: str
    def __init__(self, model_versions: _Optional[_Iterable[_Union[ModelVersion, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...

class GenerateTemporaryModelVersionCredentialsRequest(_message.Message):
    __slots__ = ("name", "version", "operation")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    operation: ModelVersionOperation
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., operation: _Optional[_Union[ModelVersionOperation, str]] = ...) -> None: ...

class GenerateTemporaryModelVersionCredentialsResponse(_message.Message):
    __slots__ = ("credentials",)
    CREDENTIALS_FIELD_NUMBER: _ClassVar[int]
    credentials: TemporaryCredentials
    def __init__(self, credentials: _Optional[_Union[TemporaryCredentials, _Mapping]] = ...) -> None: ...

class GetModelVersionDownloadUriRequest(_message.Message):
    __slots__ = ("name", "version")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class GetModelVersionDownloadUriResponse(_message.Message):
    __slots__ = ("artifact_uri",)
    ARTIFACT_URI_FIELD_NUMBER: _ClassVar[int]
    artifact_uri: str
    def __init__(self, artifact_uri: _Optional[str] = ...) -> None: ...

class FinalizeModelVersionRequest(_message.Message):
    __slots__ = ("name", "version")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class FinalizeModelVersionResponse(_message.Message):
    __slots__ = ("model_version",)
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    model_version: ModelVersion
    def __init__(self, model_version: _Optional[_Union[ModelVersion, _Mapping]] = ...) -> None: ...

class SetRegisteredModelAliasRequest(_message.Message):
    __slots__ = ("name", "alias", "version")
    NAME_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    name: str
    alias: str
    version: str
    def __init__(self, name: _Optional[str] = ..., alias: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class SetRegisteredModelAliasResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DeleteRegisteredModelAliasRequest(_message.Message):
    __slots__ = ("name", "alias")
    NAME_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    name: str
    alias: str
    def __init__(self, name: _Optional[str] = ..., alias: _Optional[str] = ...) -> None: ...

class DeleteRegisteredModelAliasResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SetRegisteredModelTagRequest(_message.Message):
    __slots__ = ("name", "key", "value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    key: str
    value: str
    def __init__(self, name: _Optional[str] = ..., key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class SetRegisteredModelTagResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DeleteRegisteredModelTagRequest(_message.Message):
    __slots__ = ("name", "key")
    NAME_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    name: str
    key: str
    def __init__(self, name: _Optional[str] = ..., key: _Optional[str] = ...) -> None: ...

class DeleteRegisteredModelTagResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SetModelVersionTagRequest(_message.Message):
    __slots__ = ("name", "version", "key", "value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    key: str
    value: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class SetModelVersionTagResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DeleteModelVersionTagRequest(_message.Message):
    __slots__ = ("name", "version", "key")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    key: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., key: _Optional[str] = ...) -> None: ...

class DeleteModelVersionTagResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetModelVersionByAliasRequest(_message.Message):
    __slots__ = ("name", "alias")
    NAME_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    name: str
    alias: str
    def __init__(self, name: _Optional[str] = ..., alias: _Optional[str] = ...) -> None: ...

class GetModelVersionByAliasResponse(_message.Message):
    __slots__ = ("model_version",)
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    model_version: ModelVersion
    def __init__(self, model_version: _Optional[_Union[ModelVersion, _Mapping]] = ...) -> None: ...

class Entity(_message.Message):
    __slots__ = ("job", "notebook", "pipeline")
    JOB_FIELD_NUMBER: _ClassVar[int]
    NOTEBOOK_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_FIELD_NUMBER: _ClassVar[int]
    job: Job
    notebook: Notebook
    pipeline: Pipeline
    def __init__(self, job: _Optional[_Union[Job, _Mapping]] = ..., notebook: _Optional[_Union[Notebook, _Mapping]] = ..., pipeline: _Optional[_Union[Pipeline, _Mapping]] = ...) -> None: ...

class Pipeline(_message.Message):
    __slots__ = ("pipeline_id", "pipeline_update_id")
    PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_UPDATE_ID_FIELD_NUMBER: _ClassVar[int]
    pipeline_id: str
    pipeline_update_id: str
    def __init__(self, pipeline_id: _Optional[str] = ..., pipeline_update_id: _Optional[str] = ...) -> None: ...

class Job(_message.Message):
    __slots__ = ("id", "task_key", "job_run_id", "task_run_id")
    ID_FIELD_NUMBER: _ClassVar[int]
    TASK_KEY_FIELD_NUMBER: _ClassVar[int]
    JOB_RUN_ID_FIELD_NUMBER: _ClassVar[int]
    TASK_RUN_ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    task_key: str
    job_run_id: str
    task_run_id: str
    def __init__(self, id: _Optional[str] = ..., task_key: _Optional[str] = ..., job_run_id: _Optional[str] = ..., task_run_id: _Optional[str] = ...) -> None: ...

class Notebook(_message.Message):
    __slots__ = ("id", "command_id", "notebook_run_id")
    ID_FIELD_NUMBER: _ClassVar[int]
    COMMAND_ID_FIELD_NUMBER: _ClassVar[int]
    NOTEBOOK_RUN_ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    command_id: str
    notebook_run_id: str
    def __init__(self, id: _Optional[str] = ..., command_id: _Optional[str] = ..., notebook_run_id: _Optional[str] = ...) -> None: ...

class Table(_message.Message):
    __slots__ = ("name", "table_type", "location", "table_id")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TABLE_TYPE_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    TABLE_ID_FIELD_NUMBER: _ClassVar[int]
    name: str
    table_type: TableType
    location: str
    table_id: str
    def __init__(self, name: _Optional[str] = ..., table_type: _Optional[_Union[TableType, str]] = ..., location: _Optional[str] = ..., table_id: _Optional[str] = ...) -> None: ...

class Securable(_message.Message):
    __slots__ = ("table",)
    TABLE_FIELD_NUMBER: _ClassVar[int]
    table: Table
    def __init__(self, table: _Optional[_Union[Table, _Mapping]] = ...) -> None: ...

class Lineage(_message.Message):
    __slots__ = ("target_securable", "source_securables")
    TARGET_SECURABLE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_SECURABLES_FIELD_NUMBER: _ClassVar[int]
    target_securable: Securable
    source_securables: _containers.RepeatedCompositeFieldContainer[Securable]
    def __init__(self, target_securable: _Optional[_Union[Securable, _Mapping]] = ..., source_securables: _Optional[_Iterable[_Union[Securable, _Mapping]]] = ...) -> None: ...

class LineageHeaderInfo(_message.Message):
    __slots__ = ("entities", "lineages")
    ENTITIES_FIELD_NUMBER: _ClassVar[int]
    LINEAGES_FIELD_NUMBER: _ClassVar[int]
    entities: _containers.RepeatedCompositeFieldContainer[Entity]
    lineages: _containers.RepeatedCompositeFieldContainer[Lineage]
    def __init__(self, entities: _Optional[_Iterable[_Union[Entity, _Mapping]]] = ..., lineages: _Optional[_Iterable[_Union[Lineage, _Mapping]]] = ...) -> None: ...

class ModelVersionLineageInfo(_message.Message):
    __slots__ = ("entities", "direction")
    ENTITIES_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    entities: _containers.RepeatedCompositeFieldContainer[Entity]
    direction: ModelVersionLineageDirection
    def __init__(self, entities: _Optional[_Iterable[_Union[Entity, _Mapping]]] = ..., direction: _Optional[_Union[ModelVersionLineageDirection, str]] = ...) -> None: ...

class EmitModelVersionLineageRequest(_message.Message):
    __slots__ = ("name", "version", "model_version_lineage_info", "securable")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_LINEAGE_INFO_FIELD_NUMBER: _ClassVar[int]
    SECURABLE_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    model_version_lineage_info: ModelVersionLineageInfo
    securable: Securable
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., model_version_lineage_info: _Optional[_Union[ModelVersionLineageInfo, _Mapping]] = ..., securable: _Optional[_Union[Securable, _Mapping]] = ...) -> None: ...

class EmitModelVersionLineageResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class IsDatabricksSdkModelsArtifactRepositoryEnabledRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class IsDatabricksSdkModelsArtifactRepositoryEnabledResponse(_message.Message):
    __slots__ = ("is_databricks_sdk_models_artifact_repository_enabled",)
    IS_DATABRICKS_SDK_MODELS_ARTIFACT_REPOSITORY_ENABLED_FIELD_NUMBER: _ClassVar[int]
    is_databricks_sdk_models_artifact_repository_enabled: bool
    def __init__(self, is_databricks_sdk_models_artifact_repository_enabled: bool = ...) -> None: ...

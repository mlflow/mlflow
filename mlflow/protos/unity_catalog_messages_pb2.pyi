import databricks_pb2 as _databricks_pb2
from scalapb import scalapb_pb2 as _scalapb_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ModelVersionStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MODEL_VERSION_STATUS_UNKNOWN: _ClassVar[ModelVersionStatus]
    PENDING_REGISTRATION: _ClassVar[ModelVersionStatus]
    FAILED_REGISTRATION: _ClassVar[ModelVersionStatus]
    READY: _ClassVar[ModelVersionStatus]

class ModelVersionOperation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNKNOWN_MODEL_VERSION_OPERATION: _ClassVar[ModelVersionOperation]
    READ_MODEL_VERSION: _ClassVar[ModelVersionOperation]
    READ_WRITE_MODEL_VERSION: _ClassVar[ModelVersionOperation]

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
MODEL_VERSION_STATUS_UNKNOWN: ModelVersionStatus
PENDING_REGISTRATION: ModelVersionStatus
FAILED_REGISTRATION: ModelVersionStatus
READY: ModelVersionStatus
UNKNOWN_MODEL_VERSION_OPERATION: ModelVersionOperation
READ_MODEL_VERSION: ModelVersionOperation
READ_WRITE_MODEL_VERSION: ModelVersionOperation
STORAGE_MODE_UNSPECIFIED: StorageMode
CUSTOMER_HOSTED: StorageMode
DEFAULT_STORAGE: StorageMode
SSE_ENCRYPTION_ALGORITHM_UNSPECIFIED: SseEncryptionAlgorithm
AWS_SSE_S3: SseEncryptionAlgorithm
AWS_SSE_KMS: SseEncryptionAlgorithm

class RegisteredModelInfo(_message.Message):
    __slots__ = ("name", "catalog_name", "schema_name", "comment", "storage_location", "full_name", "created_at", "created_by", "updated_at", "updated_by", "id", "browse_only", "owner", "metastore_id", "aliases", "tags", "deployment_job_id", "deployment_job_state")
    NAME_FIELD_NUMBER: _ClassVar[int]
    CATALOG_NAME_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_NAME_FIELD_NUMBER: _ClassVar[int]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    STORAGE_LOCATION_FIELD_NUMBER: _ClassVar[int]
    FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    CREATED_BY_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_BY_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    BROWSE_ONLY_FIELD_NUMBER: _ClassVar[int]
    OWNER_FIELD_NUMBER: _ClassVar[int]
    METASTORE_ID_FIELD_NUMBER: _ClassVar[int]
    ALIASES_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_JOB_STATE_FIELD_NUMBER: _ClassVar[int]
    name: str
    catalog_name: str
    schema_name: str
    comment: str
    storage_location: str
    full_name: str
    created_at: int
    created_by: str
    updated_at: int
    updated_by: str
    id: str
    browse_only: bool
    owner: str
    metastore_id: str
    aliases: _containers.RepeatedCompositeFieldContainer[RegisteredModelAliasInfo]
    tags: _containers.RepeatedCompositeFieldContainer[TagKeyValue]
    deployment_job_id: str
    deployment_job_state: DeploymentJobConnection.State
    def __init__(self, name: _Optional[str] = ..., catalog_name: _Optional[str] = ..., schema_name: _Optional[str] = ..., comment: _Optional[str] = ..., storage_location: _Optional[str] = ..., full_name: _Optional[str] = ..., created_at: _Optional[int] = ..., created_by: _Optional[str] = ..., updated_at: _Optional[int] = ..., updated_by: _Optional[str] = ..., id: _Optional[str] = ..., browse_only: bool = ..., owner: _Optional[str] = ..., metastore_id: _Optional[str] = ..., aliases: _Optional[_Iterable[_Union[RegisteredModelAliasInfo, _Mapping]]] = ..., tags: _Optional[_Iterable[_Union[TagKeyValue, _Mapping]]] = ..., deployment_job_id: _Optional[str] = ..., deployment_job_state: _Optional[_Union[DeploymentJobConnection.State, str]] = ...) -> None: ...

class CreateRegisteredModel(_message.Message):
    __slots__ = ("name", "catalog_name", "schema_name", "comment", "storage_location", "tags", "deployment_job_id")
    class Response(_message.Message):
        __slots__ = ("registered_model_info",)
        REGISTERED_MODEL_INFO_FIELD_NUMBER: _ClassVar[int]
        registered_model_info: RegisteredModelInfo
        def __init__(self, registered_model_info: _Optional[_Union[RegisteredModelInfo, _Mapping]] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    CATALOG_NAME_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_NAME_FIELD_NUMBER: _ClassVar[int]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    STORAGE_LOCATION_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    name: str
    catalog_name: str
    schema_name: str
    comment: str
    storage_location: str
    tags: _containers.RepeatedCompositeFieldContainer[TagKeyValue]
    deployment_job_id: str
    def __init__(self, name: _Optional[str] = ..., catalog_name: _Optional[str] = ..., schema_name: _Optional[str] = ..., comment: _Optional[str] = ..., storage_location: _Optional[str] = ..., tags: _Optional[_Iterable[_Union[TagKeyValue, _Mapping]]] = ..., deployment_job_id: _Optional[str] = ...) -> None: ...

class DeleteRegisteredModel(_message.Message):
    __slots__ = ("full_name", "force")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    FORCE_FIELD_NUMBER: _ClassVar[int]
    full_name: str
    force: bool
    def __init__(self, full_name: _Optional[str] = ..., force: bool = ...) -> None: ...

class GetRegisteredModel(_message.Message):
    __slots__ = ("full_name", "include_aliases", "include_browse")
    class Response(_message.Message):
        __slots__ = ("registered_model_info",)
        REGISTERED_MODEL_INFO_FIELD_NUMBER: _ClassVar[int]
        registered_model_info: RegisteredModelInfo
        def __init__(self, registered_model_info: _Optional[_Union[RegisteredModelInfo, _Mapping]] = ...) -> None: ...
    FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_ALIASES_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_BROWSE_FIELD_NUMBER: _ClassVar[int]
    full_name: str
    include_aliases: bool
    include_browse: bool
    def __init__(self, full_name: _Optional[str] = ..., include_aliases: bool = ..., include_browse: bool = ...) -> None: ...

class UpdateRegisteredModel(_message.Message):
    __slots__ = ("full_name", "new_name", "comment", "deployment_job_id")
    class Response(_message.Message):
        __slots__ = ("registered_model_info",)
        REGISTERED_MODEL_INFO_FIELD_NUMBER: _ClassVar[int]
        registered_model_info: RegisteredModelInfo
        def __init__(self, registered_model_info: _Optional[_Union[RegisteredModelInfo, _Mapping]] = ...) -> None: ...
    FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    NEW_NAME_FIELD_NUMBER: _ClassVar[int]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    full_name: str
    new_name: str
    comment: str
    deployment_job_id: str
    def __init__(self, full_name: _Optional[str] = ..., new_name: _Optional[str] = ..., comment: _Optional[str] = ..., deployment_job_id: _Optional[str] = ...) -> None: ...

class ListRegisteredModels(_message.Message):
    __slots__ = ("catalog_name", "schema_name", "include_browse", "max_results", "page_token")
    class Response(_message.Message):
        __slots__ = ("registered_models", "next_page_token")
        REGISTERED_MODELS_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        registered_models: _containers.RepeatedCompositeFieldContainer[RegisteredModelInfo]
        next_page_token: str
        def __init__(self, registered_models: _Optional[_Iterable[_Union[RegisteredModelInfo, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    CATALOG_NAME_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_NAME_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_BROWSE_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    catalog_name: str
    schema_name: str
    include_browse: bool
    max_results: int
    page_token: str
    def __init__(self, catalog_name: _Optional[str] = ..., schema_name: _Optional[str] = ..., include_browse: bool = ..., max_results: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...

class ModelVersionInfo(_message.Message):
    __slots__ = ("model_name", "catalog_name", "schema_name", "source", "comment", "run_id", "status", "version", "storage_location", "created_at", "created_by", "updated_at", "updated_by", "id", "run_workspace_id", "metastore_id", "aliases", "tags", "model_version_dependencies", "model_id", "model_params", "model_metrics", "deployment_job_state")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    CATALOG_NAME_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_NAME_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    STORAGE_LOCATION_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    CREATED_BY_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_BY_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    RUN_WORKSPACE_ID_FIELD_NUMBER: _ClassVar[int]
    METASTORE_ID_FIELD_NUMBER: _ClassVar[int]
    ALIASES_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_DEPENDENCIES_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_PARAMS_FIELD_NUMBER: _ClassVar[int]
    MODEL_METRICS_FIELD_NUMBER: _ClassVar[int]
    DEPLOYMENT_JOB_STATE_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    catalog_name: str
    schema_name: str
    source: str
    comment: str
    run_id: str
    status: ModelVersionStatus
    version: int
    storage_location: str
    created_at: int
    created_by: str
    updated_at: int
    updated_by: str
    id: str
    run_workspace_id: int
    metastore_id: str
    aliases: _containers.RepeatedCompositeFieldContainer[RegisteredModelAliasInfo]
    tags: _containers.RepeatedCompositeFieldContainer[TagKeyValue]
    model_version_dependencies: DependencyList
    model_id: str
    model_params: _containers.RepeatedCompositeFieldContainer[ModelParam]
    model_metrics: _containers.RepeatedCompositeFieldContainer[ModelMetric]
    deployment_job_state: ModelVersionDeploymentJobState
    def __init__(self, model_name: _Optional[str] = ..., catalog_name: _Optional[str] = ..., schema_name: _Optional[str] = ..., source: _Optional[str] = ..., comment: _Optional[str] = ..., run_id: _Optional[str] = ..., status: _Optional[_Union[ModelVersionStatus, str]] = ..., version: _Optional[int] = ..., storage_location: _Optional[str] = ..., created_at: _Optional[int] = ..., created_by: _Optional[str] = ..., updated_at: _Optional[int] = ..., updated_by: _Optional[str] = ..., id: _Optional[str] = ..., run_workspace_id: _Optional[int] = ..., metastore_id: _Optional[str] = ..., aliases: _Optional[_Iterable[_Union[RegisteredModelAliasInfo, _Mapping]]] = ..., tags: _Optional[_Iterable[_Union[TagKeyValue, _Mapping]]] = ..., model_version_dependencies: _Optional[_Union[DependencyList, _Mapping]] = ..., model_id: _Optional[str] = ..., model_params: _Optional[_Iterable[_Union[ModelParam, _Mapping]]] = ..., model_metrics: _Optional[_Iterable[_Union[ModelMetric, _Mapping]]] = ..., deployment_job_state: _Optional[_Union[ModelVersionDeploymentJobState, _Mapping]] = ...) -> None: ...

class CreateModelVersion(_message.Message):
    __slots__ = ("model_name", "catalog_name", "schema_name", "source", "run_id", "comment", "tags", "model_version_dependencies", "model_id", "feature_deps", "run_tracking_server_id")
    class Response(_message.Message):
        __slots__ = ("model_version_info",)
        MODEL_VERSION_INFO_FIELD_NUMBER: _ClassVar[int]
        model_version_info: ModelVersionInfo
        def __init__(self, model_version_info: _Optional[_Union[ModelVersionInfo, _Mapping]] = ...) -> None: ...
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    CATALOG_NAME_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_NAME_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_DEPENDENCIES_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    FEATURE_DEPS_FIELD_NUMBER: _ClassVar[int]
    RUN_TRACKING_SERVER_ID_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    catalog_name: str
    schema_name: str
    source: str
    run_id: str
    comment: str
    tags: _containers.RepeatedCompositeFieldContainer[TagKeyValue]
    model_version_dependencies: DependencyList
    model_id: str
    feature_deps: str
    run_tracking_server_id: str
    def __init__(self, model_name: _Optional[str] = ..., catalog_name: _Optional[str] = ..., schema_name: _Optional[str] = ..., source: _Optional[str] = ..., run_id: _Optional[str] = ..., comment: _Optional[str] = ..., tags: _Optional[_Iterable[_Union[TagKeyValue, _Mapping]]] = ..., model_version_dependencies: _Optional[_Union[DependencyList, _Mapping]] = ..., model_id: _Optional[str] = ..., feature_deps: _Optional[str] = ..., run_tracking_server_id: _Optional[str] = ...) -> None: ...

class DeleteModelVersion(_message.Message):
    __slots__ = ("full_name", "version")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    full_name: str
    version: int
    def __init__(self, full_name: _Optional[str] = ..., version: _Optional[int] = ...) -> None: ...

class FinalizeModelVersion(_message.Message):
    __slots__ = ("full_name", "version")
    class Response(_message.Message):
        __slots__ = ("model_version_info",)
        MODEL_VERSION_INFO_FIELD_NUMBER: _ClassVar[int]
        model_version_info: ModelVersionInfo
        def __init__(self, model_version_info: _Optional[_Union[ModelVersionInfo, _Mapping]] = ...) -> None: ...
    FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    full_name: str
    version: int
    def __init__(self, full_name: _Optional[str] = ..., version: _Optional[int] = ...) -> None: ...

class GetModelVersion(_message.Message):
    __slots__ = ("full_name", "version", "include_aliases", "include_browse")
    class Response(_message.Message):
        __slots__ = ("model_version_info",)
        MODEL_VERSION_INFO_FIELD_NUMBER: _ClassVar[int]
        model_version_info: ModelVersionInfo
        def __init__(self, model_version_info: _Optional[_Union[ModelVersionInfo, _Mapping]] = ...) -> None: ...
    FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_ALIASES_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_BROWSE_FIELD_NUMBER: _ClassVar[int]
    full_name: str
    version: int
    include_aliases: bool
    include_browse: bool
    def __init__(self, full_name: _Optional[str] = ..., version: _Optional[int] = ..., include_aliases: bool = ..., include_browse: bool = ...) -> None: ...

class UpdateModelVersion(_message.Message):
    __slots__ = ("full_name", "version", "comment")
    class Response(_message.Message):
        __slots__ = ("model_version_info",)
        MODEL_VERSION_INFO_FIELD_NUMBER: _ClassVar[int]
        model_version_info: ModelVersionInfo
        def __init__(self, model_version_info: _Optional[_Union[ModelVersionInfo, _Mapping]] = ...) -> None: ...
    FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    full_name: str
    version: int
    comment: str
    def __init__(self, full_name: _Optional[str] = ..., version: _Optional[int] = ..., comment: _Optional[str] = ...) -> None: ...

class ListModelVersions(_message.Message):
    __slots__ = ("full_name", "max_results", "page_token", "include_browse")
    class Response(_message.Message):
        __slots__ = ("model_versions", "next_page_token")
        MODEL_VERSIONS_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        model_versions: _containers.RepeatedCompositeFieldContainer[ModelVersionInfo]
        next_page_token: str
        def __init__(self, model_versions: _Optional[_Iterable[_Union[ModelVersionInfo, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_BROWSE_FIELD_NUMBER: _ClassVar[int]
    full_name: str
    max_results: int
    page_token: str
    include_browse: bool
    def __init__(self, full_name: _Optional[str] = ..., max_results: _Optional[int] = ..., page_token: _Optional[str] = ..., include_browse: bool = ...) -> None: ...

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

class GenerateTemporaryModelVersionCredential(_message.Message):
    __slots__ = ("catalog_name", "schema_name", "model_name", "version", "operation")
    class Response(_message.Message):
        __slots__ = ("credentials",)
        CREDENTIALS_FIELD_NUMBER: _ClassVar[int]
        credentials: TemporaryCredentials
        def __init__(self, credentials: _Optional[_Union[TemporaryCredentials, _Mapping]] = ...) -> None: ...
    CATALOG_NAME_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    catalog_name: str
    schema_name: str
    model_name: str
    version: int
    operation: ModelVersionOperation
    def __init__(self, catalog_name: _Optional[str] = ..., schema_name: _Optional[str] = ..., model_name: _Optional[str] = ..., version: _Optional[int] = ..., operation: _Optional[_Union[ModelVersionOperation, str]] = ...) -> None: ...

class TagKeyValue(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class RegisteredModelAliasInfo(_message.Message):
    __slots__ = ("alias_name", "version_num", "id", "model_name", "catalog_name", "schema_name")
    ALIAS_NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_NUM_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    CATALOG_NAME_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_NAME_FIELD_NUMBER: _ClassVar[int]
    alias_name: str
    version_num: int
    id: str
    model_name: str
    catalog_name: str
    schema_name: str
    def __init__(self, alias_name: _Optional[str] = ..., version_num: _Optional[int] = ..., id: _Optional[str] = ..., model_name: _Optional[str] = ..., catalog_name: _Optional[str] = ..., schema_name: _Optional[str] = ...) -> None: ...

class TableDependency(_message.Message):
    __slots__ = ("table_full_name",)
    TABLE_FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    table_full_name: str
    def __init__(self, table_full_name: _Optional[str] = ...) -> None: ...

class FunctionDependency(_message.Message):
    __slots__ = ("function_full_name",)
    FUNCTION_FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    function_full_name: str
    def __init__(self, function_full_name: _Optional[str] = ...) -> None: ...

class ConnectionDependency(_message.Message):
    __slots__ = ("connection_name",)
    CONNECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    connection_name: str
    def __init__(self, connection_name: _Optional[str] = ...) -> None: ...

class ModelVersionDependency(_message.Message):
    __slots__ = ("table", "function", "connection")
    TABLE_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_FIELD_NUMBER: _ClassVar[int]
    CONNECTION_FIELD_NUMBER: _ClassVar[int]
    table: TableDependency
    function: FunctionDependency
    connection: ConnectionDependency
    def __init__(self, table: _Optional[_Union[TableDependency, _Mapping]] = ..., function: _Optional[_Union[FunctionDependency, _Mapping]] = ..., connection: _Optional[_Union[ConnectionDependency, _Mapping]] = ...) -> None: ...

class DependencyList(_message.Message):
    __slots__ = ("dependencies",)
    DEPENDENCIES_FIELD_NUMBER: _ClassVar[int]
    dependencies: _containers.RepeatedCompositeFieldContainer[ModelVersionDependency]
    def __init__(self, dependencies: _Optional[_Iterable[_Union[ModelVersionDependency, _Mapping]]] = ...) -> None: ...

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

class SseEncryptionDetails(_message.Message):
    __slots__ = ("algorithm", "aws_kms_key_arn")
    ALGORITHM_FIELD_NUMBER: _ClassVar[int]
    AWS_KMS_KEY_ARN_FIELD_NUMBER: _ClassVar[int]
    algorithm: SseEncryptionAlgorithm
    aws_kms_key_arn: str
    def __init__(self, algorithm: _Optional[_Union[SseEncryptionAlgorithm, str]] = ..., aws_kms_key_arn: _Optional[str] = ...) -> None: ...

class EncryptionDetails(_message.Message):
    __slots__ = ("sse_encryption_details",)
    SSE_ENCRYPTION_DETAILS_FIELD_NUMBER: _ClassVar[int]
    sse_encryption_details: SseEncryptionDetails
    def __init__(self, sse_encryption_details: _Optional[_Union[SseEncryptionDetails, _Mapping]] = ...) -> None: ...

class R2Credentials(_message.Message):
    __slots__ = ("access_key_id", "secret_access_key", "session_token")
    ACCESS_KEY_ID_FIELD_NUMBER: _ClassVar[int]
    SECRET_ACCESS_KEY_FIELD_NUMBER: _ClassVar[int]
    SESSION_TOKEN_FIELD_NUMBER: _ClassVar[int]
    access_key_id: str
    secret_access_key: str
    session_token: str
    def __init__(self, access_key_id: _Optional[str] = ..., secret_access_key: _Optional[str] = ..., session_token: _Optional[str] = ...) -> None: ...

class GetModelVersionByAlias(_message.Message):
    __slots__ = ("full_name", "alias", "include_aliases")
    class Response(_message.Message):
        __slots__ = ("model_version_info",)
        MODEL_VERSION_INFO_FIELD_NUMBER: _ClassVar[int]
        model_version_info: ModelVersionInfo
        def __init__(self, model_version_info: _Optional[_Union[ModelVersionInfo, _Mapping]] = ...) -> None: ...
    FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_ALIASES_FIELD_NUMBER: _ClassVar[int]
    full_name: str
    alias: str
    include_aliases: bool
    def __init__(self, full_name: _Optional[str] = ..., alias: _Optional[str] = ..., include_aliases: bool = ...) -> None: ...

class SetRegisteredModelAlias(_message.Message):
    __slots__ = ("full_name", "alias", "version_num")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    VERSION_NUM_FIELD_NUMBER: _ClassVar[int]
    full_name: str
    alias: str
    version_num: int
    def __init__(self, full_name: _Optional[str] = ..., alias: _Optional[str] = ..., version_num: _Optional[int] = ...) -> None: ...

class DeleteRegisteredModelAlias(_message.Message):
    __slots__ = ("full_name", "alias")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    full_name: str
    alias: str
    def __init__(self, full_name: _Optional[str] = ..., alias: _Optional[str] = ...) -> None: ...

class TagAssignmentsChange(_message.Message):
    __slots__ = ("remove", "add_tags")
    REMOVE_FIELD_NUMBER: _ClassVar[int]
    ADD_TAGS_FIELD_NUMBER: _ClassVar[int]
    remove: _containers.RepeatedScalarFieldContainer[str]
    add_tags: _containers.RepeatedCompositeFieldContainer[TagKeyValue]
    def __init__(self, remove: _Optional[_Iterable[str]] = ..., add_tags: _Optional[_Iterable[_Union[TagKeyValue, _Mapping]]] = ...) -> None: ...

class UpdateTagSecurableAssignments(_message.Message):
    __slots__ = ("changes", "securable_type", "securable_full_name")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    CHANGES_FIELD_NUMBER: _ClassVar[int]
    SECURABLE_TYPE_FIELD_NUMBER: _ClassVar[int]
    SECURABLE_FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    changes: TagAssignmentsChange
    securable_type: str
    securable_full_name: str
    def __init__(self, changes: _Optional[_Union[TagAssignmentsChange, _Mapping]] = ..., securable_type: _Optional[str] = ..., securable_full_name: _Optional[str] = ...) -> None: ...

class UpdateTagSubentityAssignments(_message.Message):
    __slots__ = ("changes", "securable_type", "securable_full_name", "subentity_name")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    CHANGES_FIELD_NUMBER: _ClassVar[int]
    SECURABLE_TYPE_FIELD_NUMBER: _ClassVar[int]
    SECURABLE_FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    SUBENTITY_NAME_FIELD_NUMBER: _ClassVar[int]
    changes: TagAssignmentsChange
    securable_type: str
    securable_full_name: str
    subentity_name: str
    def __init__(self, changes: _Optional[_Union[TagAssignmentsChange, _Mapping]] = ..., securable_type: _Optional[str] = ..., securable_full_name: _Optional[str] = ..., subentity_name: _Optional[str] = ...) -> None: ...

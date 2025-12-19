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
MODEL_VERSION_STATUS_UNKNOWN: ModelVersionStatus
PENDING_REGISTRATION: ModelVersionStatus
FAILED_REGISTRATION: ModelVersionStatus
READY: ModelVersionStatus
UNKNOWN_MODEL_VERSION_OPERATION: ModelVersionOperation
READ_MODEL_VERSION: ModelVersionOperation
READ_WRITE_MODEL_VERSION: ModelVersionOperation

class RegisteredModelInfo(_message.Message):
    __slots__ = ("name", "catalog_name", "schema_name", "comment", "storage_location", "full_name", "created_at", "created_by", "updated_at", "updated_by", "id", "browse_only")
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
    def __init__(self, name: _Optional[str] = ..., catalog_name: _Optional[str] = ..., schema_name: _Optional[str] = ..., comment: _Optional[str] = ..., storage_location: _Optional[str] = ..., full_name: _Optional[str] = ..., created_at: _Optional[int] = ..., created_by: _Optional[str] = ..., updated_at: _Optional[int] = ..., updated_by: _Optional[str] = ..., id: _Optional[str] = ..., browse_only: bool = ...) -> None: ...

class CreateRegisteredModel(_message.Message):
    __slots__ = ("name", "catalog_name", "schema_name", "comment", "storage_location")
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
    name: str
    catalog_name: str
    schema_name: str
    comment: str
    storage_location: str
    def __init__(self, name: _Optional[str] = ..., catalog_name: _Optional[str] = ..., schema_name: _Optional[str] = ..., comment: _Optional[str] = ..., storage_location: _Optional[str] = ...) -> None: ...

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
    __slots__ = ("full_name",)
    class Response(_message.Message):
        __slots__ = ("registered_model_info",)
        REGISTERED_MODEL_INFO_FIELD_NUMBER: _ClassVar[int]
        registered_model_info: RegisteredModelInfo
        def __init__(self, registered_model_info: _Optional[_Union[RegisteredModelInfo, _Mapping]] = ...) -> None: ...
    FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    full_name: str
    def __init__(self, full_name: _Optional[str] = ...) -> None: ...

class UpdateRegisteredModel(_message.Message):
    __slots__ = ("full_name", "new_name", "comment")
    class Response(_message.Message):
        __slots__ = ("registered_model_info",)
        REGISTERED_MODEL_INFO_FIELD_NUMBER: _ClassVar[int]
        registered_model_info: RegisteredModelInfo
        def __init__(self, registered_model_info: _Optional[_Union[RegisteredModelInfo, _Mapping]] = ...) -> None: ...
    FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    NEW_NAME_FIELD_NUMBER: _ClassVar[int]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    full_name: str
    new_name: str
    comment: str
    def __init__(self, full_name: _Optional[str] = ..., new_name: _Optional[str] = ..., comment: _Optional[str] = ...) -> None: ...

class ListRegisteredModels(_message.Message):
    __slots__ = ("catalog_name", "schema_name", "max_results", "page_token")
    class Response(_message.Message):
        __slots__ = ("registered_models", "next_page_token")
        REGISTERED_MODELS_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        registered_models: _containers.RepeatedCompositeFieldContainer[RegisteredModelInfo]
        next_page_token: str
        def __init__(self, registered_models: _Optional[_Iterable[_Union[RegisteredModelInfo, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    CATALOG_NAME_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_NAME_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    catalog_name: str
    schema_name: str
    max_results: int
    page_token: str
    def __init__(self, catalog_name: _Optional[str] = ..., schema_name: _Optional[str] = ..., max_results: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...

class ModelVersionInfo(_message.Message):
    __slots__ = ("model_name", "catalog_name", "schema_name", "source", "comment", "run_id", "status", "version", "storage_location", "created_at", "created_by", "updated_at", "updated_by", "id")
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
    def __init__(self, model_name: _Optional[str] = ..., catalog_name: _Optional[str] = ..., schema_name: _Optional[str] = ..., source: _Optional[str] = ..., comment: _Optional[str] = ..., run_id: _Optional[str] = ..., status: _Optional[_Union[ModelVersionStatus, str]] = ..., version: _Optional[int] = ..., storage_location: _Optional[str] = ..., created_at: _Optional[int] = ..., created_by: _Optional[str] = ..., updated_at: _Optional[int] = ..., updated_by: _Optional[str] = ..., id: _Optional[str] = ...) -> None: ...

class CreateModelVersion(_message.Message):
    __slots__ = ("model_name", "catalog_name", "schema_name", "source", "run_id", "comment")
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
    model_name: str
    catalog_name: str
    schema_name: str
    source: str
    run_id: str
    comment: str
    def __init__(self, model_name: _Optional[str] = ..., catalog_name: _Optional[str] = ..., schema_name: _Optional[str] = ..., source: _Optional[str] = ..., run_id: _Optional[str] = ..., comment: _Optional[str] = ...) -> None: ...

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
    __slots__ = ("full_name", "max_results", "page_token")
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
    full_name: str
    max_results: int
    page_token: str
    def __init__(self, full_name: _Optional[str] = ..., max_results: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...

class TemporaryCredentials(_message.Message):
    __slots__ = ("aws_temp_credentials", "azure_user_delegation_sas", "gcp_oauth_token", "expiration_time")
    AWS_TEMP_CREDENTIALS_FIELD_NUMBER: _ClassVar[int]
    AZURE_USER_DELEGATION_SAS_FIELD_NUMBER: _ClassVar[int]
    GCP_OAUTH_TOKEN_FIELD_NUMBER: _ClassVar[int]
    EXPIRATION_TIME_FIELD_NUMBER: _ClassVar[int]
    aws_temp_credentials: AwsCredentials
    azure_user_delegation_sas: AzureUserDelegationSAS
    gcp_oauth_token: GcpOauthToken
    expiration_time: int
    def __init__(self, aws_temp_credentials: _Optional[_Union[AwsCredentials, _Mapping]] = ..., azure_user_delegation_sas: _Optional[_Union[AzureUserDelegationSAS, _Mapping]] = ..., gcp_oauth_token: _Optional[_Union[GcpOauthToken, _Mapping]] = ..., expiration_time: _Optional[int] = ...) -> None: ...

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

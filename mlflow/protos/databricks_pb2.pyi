from google.protobuf import descriptor_pb2 as _descriptor_pb2
from scalapb import scalapb_pb2 as _scalapb_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Visibility(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PUBLIC: _ClassVar[Visibility]
    INTERNAL: _ClassVar[Visibility]
    PUBLIC_UNDOCUMENTED: _ClassVar[Visibility]

class ErrorCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    INTERNAL_ERROR: _ClassVar[ErrorCode]
    TEMPORARILY_UNAVAILABLE: _ClassVar[ErrorCode]
    IO_ERROR: _ClassVar[ErrorCode]
    BAD_REQUEST: _ClassVar[ErrorCode]
    SERVICE_UNDER_MAINTENANCE: _ClassVar[ErrorCode]
    WORKSPACE_TEMPORARILY_UNAVAILABLE: _ClassVar[ErrorCode]
    DEADLINE_EXCEEDED: _ClassVar[ErrorCode]
    CANCELLED: _ClassVar[ErrorCode]
    RESOURCE_EXHAUSTED: _ClassVar[ErrorCode]
    ABORTED: _ClassVar[ErrorCode]
    NOT_FOUND: _ClassVar[ErrorCode]
    ALREADY_EXISTS: _ClassVar[ErrorCode]
    UNAUTHENTICATED: _ClassVar[ErrorCode]
    INVALID_PARAMETER_VALUE: _ClassVar[ErrorCode]
    ENDPOINT_NOT_FOUND: _ClassVar[ErrorCode]
    MALFORMED_REQUEST: _ClassVar[ErrorCode]
    INVALID_STATE: _ClassVar[ErrorCode]
    PERMISSION_DENIED: _ClassVar[ErrorCode]
    FEATURE_DISABLED: _ClassVar[ErrorCode]
    CUSTOMER_UNAUTHORIZED: _ClassVar[ErrorCode]
    REQUEST_LIMIT_EXCEEDED: _ClassVar[ErrorCode]
    RESOURCE_CONFLICT: _ClassVar[ErrorCode]
    UNPARSEABLE_HTTP_ERROR: _ClassVar[ErrorCode]
    NOT_IMPLEMENTED: _ClassVar[ErrorCode]
    DATA_LOSS: _ClassVar[ErrorCode]
    INVALID_STATE_TRANSITION: _ClassVar[ErrorCode]
    COULD_NOT_ACQUIRE_LOCK: _ClassVar[ErrorCode]
    RESOURCE_ALREADY_EXISTS: _ClassVar[ErrorCode]
    RESOURCE_DOES_NOT_EXIST: _ClassVar[ErrorCode]
    QUOTA_EXCEEDED: _ClassVar[ErrorCode]
    MAX_BLOCK_SIZE_EXCEEDED: _ClassVar[ErrorCode]
    MAX_READ_SIZE_EXCEEDED: _ClassVar[ErrorCode]
    PARTIAL_DELETE: _ClassVar[ErrorCode]
    MAX_LIST_SIZE_EXCEEDED: _ClassVar[ErrorCode]
    DRY_RUN_FAILED: _ClassVar[ErrorCode]
    RESOURCE_LIMIT_EXCEEDED: _ClassVar[ErrorCode]
    DIRECTORY_NOT_EMPTY: _ClassVar[ErrorCode]
    DIRECTORY_PROTECTED: _ClassVar[ErrorCode]
    MAX_NOTEBOOK_SIZE_EXCEEDED: _ClassVar[ErrorCode]
    MAX_CHILD_NODE_SIZE_EXCEEDED: _ClassVar[ErrorCode]
    SEARCH_QUERY_TOO_LONG: _ClassVar[ErrorCode]
    SEARCH_QUERY_TOO_SHORT: _ClassVar[ErrorCode]
    MANAGED_RESOURCE_GROUP_DOES_NOT_EXIST: _ClassVar[ErrorCode]
    PERMISSION_NOT_PROPAGATED: _ClassVar[ErrorCode]
    DEPLOYMENT_TIMEOUT: _ClassVar[ErrorCode]
    GIT_CONFLICT: _ClassVar[ErrorCode]
    GIT_UNKNOWN_REF: _ClassVar[ErrorCode]
    GIT_SENSITIVE_TOKEN_DETECTED: _ClassVar[ErrorCode]
    GIT_URL_NOT_ON_ALLOW_LIST: _ClassVar[ErrorCode]
    GIT_REMOTE_ERROR: _ClassVar[ErrorCode]
    PROJECTS_OPERATION_TIMEOUT: _ClassVar[ErrorCode]
    IPYNB_FILE_IN_REPO: _ClassVar[ErrorCode]
    INSECURE_PARTNER_RESPONSE: _ClassVar[ErrorCode]
    MALFORMED_PARTNER_RESPONSE: _ClassVar[ErrorCode]
    METASTORE_DOES_NOT_EXIST: _ClassVar[ErrorCode]
    DAC_DOES_NOT_EXIST: _ClassVar[ErrorCode]
    CATALOG_DOES_NOT_EXIST: _ClassVar[ErrorCode]
    SCHEMA_DOES_NOT_EXIST: _ClassVar[ErrorCode]
    TABLE_DOES_NOT_EXIST: _ClassVar[ErrorCode]
    SHARE_DOES_NOT_EXIST: _ClassVar[ErrorCode]
    RECIPIENT_DOES_NOT_EXIST: _ClassVar[ErrorCode]
    STORAGE_CREDENTIAL_DOES_NOT_EXIST: _ClassVar[ErrorCode]
    EXTERNAL_LOCATION_DOES_NOT_EXIST: _ClassVar[ErrorCode]
    PRINCIPAL_DOES_NOT_EXIST: _ClassVar[ErrorCode]
    PROVIDER_DOES_NOT_EXIST: _ClassVar[ErrorCode]
    METASTORE_ALREADY_EXISTS: _ClassVar[ErrorCode]
    DAC_ALREADY_EXISTS: _ClassVar[ErrorCode]
    CATALOG_ALREADY_EXISTS: _ClassVar[ErrorCode]
    SCHEMA_ALREADY_EXISTS: _ClassVar[ErrorCode]
    TABLE_ALREADY_EXISTS: _ClassVar[ErrorCode]
    SHARE_ALREADY_EXISTS: _ClassVar[ErrorCode]
    RECIPIENT_ALREADY_EXISTS: _ClassVar[ErrorCode]
    STORAGE_CREDENTIAL_ALREADY_EXISTS: _ClassVar[ErrorCode]
    EXTERNAL_LOCATION_ALREADY_EXISTS: _ClassVar[ErrorCode]
    PROVIDER_ALREADY_EXISTS: _ClassVar[ErrorCode]
    CATALOG_NOT_EMPTY: _ClassVar[ErrorCode]
    SCHEMA_NOT_EMPTY: _ClassVar[ErrorCode]
    METASTORE_NOT_EMPTY: _ClassVar[ErrorCode]
    PROVIDER_SHARE_NOT_ACCESSIBLE: _ClassVar[ErrorCode]
PUBLIC: Visibility
INTERNAL: Visibility
PUBLIC_UNDOCUMENTED: Visibility
INTERNAL_ERROR: ErrorCode
TEMPORARILY_UNAVAILABLE: ErrorCode
IO_ERROR: ErrorCode
BAD_REQUEST: ErrorCode
SERVICE_UNDER_MAINTENANCE: ErrorCode
WORKSPACE_TEMPORARILY_UNAVAILABLE: ErrorCode
DEADLINE_EXCEEDED: ErrorCode
CANCELLED: ErrorCode
RESOURCE_EXHAUSTED: ErrorCode
ABORTED: ErrorCode
NOT_FOUND: ErrorCode
ALREADY_EXISTS: ErrorCode
UNAUTHENTICATED: ErrorCode
INVALID_PARAMETER_VALUE: ErrorCode
ENDPOINT_NOT_FOUND: ErrorCode
MALFORMED_REQUEST: ErrorCode
INVALID_STATE: ErrorCode
PERMISSION_DENIED: ErrorCode
FEATURE_DISABLED: ErrorCode
CUSTOMER_UNAUTHORIZED: ErrorCode
REQUEST_LIMIT_EXCEEDED: ErrorCode
RESOURCE_CONFLICT: ErrorCode
UNPARSEABLE_HTTP_ERROR: ErrorCode
NOT_IMPLEMENTED: ErrorCode
DATA_LOSS: ErrorCode
INVALID_STATE_TRANSITION: ErrorCode
COULD_NOT_ACQUIRE_LOCK: ErrorCode
RESOURCE_ALREADY_EXISTS: ErrorCode
RESOURCE_DOES_NOT_EXIST: ErrorCode
QUOTA_EXCEEDED: ErrorCode
MAX_BLOCK_SIZE_EXCEEDED: ErrorCode
MAX_READ_SIZE_EXCEEDED: ErrorCode
PARTIAL_DELETE: ErrorCode
MAX_LIST_SIZE_EXCEEDED: ErrorCode
DRY_RUN_FAILED: ErrorCode
RESOURCE_LIMIT_EXCEEDED: ErrorCode
DIRECTORY_NOT_EMPTY: ErrorCode
DIRECTORY_PROTECTED: ErrorCode
MAX_NOTEBOOK_SIZE_EXCEEDED: ErrorCode
MAX_CHILD_NODE_SIZE_EXCEEDED: ErrorCode
SEARCH_QUERY_TOO_LONG: ErrorCode
SEARCH_QUERY_TOO_SHORT: ErrorCode
MANAGED_RESOURCE_GROUP_DOES_NOT_EXIST: ErrorCode
PERMISSION_NOT_PROPAGATED: ErrorCode
DEPLOYMENT_TIMEOUT: ErrorCode
GIT_CONFLICT: ErrorCode
GIT_UNKNOWN_REF: ErrorCode
GIT_SENSITIVE_TOKEN_DETECTED: ErrorCode
GIT_URL_NOT_ON_ALLOW_LIST: ErrorCode
GIT_REMOTE_ERROR: ErrorCode
PROJECTS_OPERATION_TIMEOUT: ErrorCode
IPYNB_FILE_IN_REPO: ErrorCode
INSECURE_PARTNER_RESPONSE: ErrorCode
MALFORMED_PARTNER_RESPONSE: ErrorCode
METASTORE_DOES_NOT_EXIST: ErrorCode
DAC_DOES_NOT_EXIST: ErrorCode
CATALOG_DOES_NOT_EXIST: ErrorCode
SCHEMA_DOES_NOT_EXIST: ErrorCode
TABLE_DOES_NOT_EXIST: ErrorCode
SHARE_DOES_NOT_EXIST: ErrorCode
RECIPIENT_DOES_NOT_EXIST: ErrorCode
STORAGE_CREDENTIAL_DOES_NOT_EXIST: ErrorCode
EXTERNAL_LOCATION_DOES_NOT_EXIST: ErrorCode
PRINCIPAL_DOES_NOT_EXIST: ErrorCode
PROVIDER_DOES_NOT_EXIST: ErrorCode
METASTORE_ALREADY_EXISTS: ErrorCode
DAC_ALREADY_EXISTS: ErrorCode
CATALOG_ALREADY_EXISTS: ErrorCode
SCHEMA_ALREADY_EXISTS: ErrorCode
TABLE_ALREADY_EXISTS: ErrorCode
SHARE_ALREADY_EXISTS: ErrorCode
RECIPIENT_ALREADY_EXISTS: ErrorCode
STORAGE_CREDENTIAL_ALREADY_EXISTS: ErrorCode
EXTERNAL_LOCATION_ALREADY_EXISTS: ErrorCode
PROVIDER_ALREADY_EXISTS: ErrorCode
CATALOG_NOT_EMPTY: ErrorCode
SCHEMA_NOT_EMPTY: ErrorCode
METASTORE_NOT_EMPTY: ErrorCode
PROVIDER_SHARE_NOT_ACCESSIBLE: ErrorCode
VISIBILITY_FIELD_NUMBER: _ClassVar[int]
visibility: _descriptor.FieldDescriptor
VALIDATE_REQUIRED_FIELD_NUMBER: _ClassVar[int]
validate_required: _descriptor.FieldDescriptor
JSON_INLINE_FIELD_NUMBER: _ClassVar[int]
json_inline: _descriptor.FieldDescriptor
JSON_MAP_FIELD_NUMBER: _ClassVar[int]
json_map: _descriptor.FieldDescriptor
FIELD_DOC_FIELD_NUMBER: _ClassVar[int]
field_doc: _descriptor.FieldDescriptor
RPC_FIELD_NUMBER: _ClassVar[int]
rpc: _descriptor.FieldDescriptor
METHOD_DOC_FIELD_NUMBER: _ClassVar[int]
method_doc: _descriptor.FieldDescriptor
GRAPHQL_FIELD_NUMBER: _ClassVar[int]
graphql: _descriptor.FieldDescriptor
MESSAGE_DOC_FIELD_NUMBER: _ClassVar[int]
message_doc: _descriptor.FieldDescriptor
SERVICE_DOC_FIELD_NUMBER: _ClassVar[int]
service_doc: _descriptor.FieldDescriptor
ENUM_DOC_FIELD_NUMBER: _ClassVar[int]
enum_doc: _descriptor.FieldDescriptor
ENUM_VALUE_VISIBILITY_FIELD_NUMBER: _ClassVar[int]
enum_value_visibility: _descriptor.FieldDescriptor
ENUM_VALUE_DOC_FIELD_NUMBER: _ClassVar[int]
enum_value_doc: _descriptor.FieldDescriptor

class DatabricksRpcOptions(_message.Message):
    __slots__ = ("endpoints", "visibility", "error_codes", "rate_limit", "rpc_doc_title")
    ENDPOINTS_FIELD_NUMBER: _ClassVar[int]
    VISIBILITY_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODES_FIELD_NUMBER: _ClassVar[int]
    RATE_LIMIT_FIELD_NUMBER: _ClassVar[int]
    RPC_DOC_TITLE_FIELD_NUMBER: _ClassVar[int]
    endpoints: _containers.RepeatedCompositeFieldContainer[HttpEndpoint]
    visibility: Visibility
    error_codes: _containers.RepeatedScalarFieldContainer[ErrorCode]
    rate_limit: RateLimit
    rpc_doc_title: str
    def __init__(self, endpoints: _Optional[_Iterable[_Union[HttpEndpoint, _Mapping]]] = ..., visibility: _Optional[_Union[Visibility, str]] = ..., error_codes: _Optional[_Iterable[_Union[ErrorCode, str]]] = ..., rate_limit: _Optional[_Union[RateLimit, _Mapping]] = ..., rpc_doc_title: _Optional[str] = ...) -> None: ...

class DatabricksGraphqlOptions(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HttpEndpoint(_message.Message):
    __slots__ = ("method", "path", "since")
    METHOD_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    SINCE_FIELD_NUMBER: _ClassVar[int]
    method: str
    path: str
    since: ApiVersion
    def __init__(self, method: _Optional[str] = ..., path: _Optional[str] = ..., since: _Optional[_Union[ApiVersion, _Mapping]] = ...) -> None: ...

class ApiVersion(_message.Message):
    __slots__ = ("major", "minor")
    MAJOR_FIELD_NUMBER: _ClassVar[int]
    MINOR_FIELD_NUMBER: _ClassVar[int]
    major: int
    minor: int
    def __init__(self, major: _Optional[int] = ..., minor: _Optional[int] = ...) -> None: ...

class RateLimit(_message.Message):
    __slots__ = ("max_burst", "max_sustained_per_second")
    MAX_BURST_FIELD_NUMBER: _ClassVar[int]
    MAX_SUSTAINED_PER_SECOND_FIELD_NUMBER: _ClassVar[int]
    max_burst: int
    max_sustained_per_second: int
    def __init__(self, max_burst: _Optional[int] = ..., max_sustained_per_second: _Optional[int] = ...) -> None: ...

class DocumentationMetadata(_message.Message):
    __slots__ = ("docstring", "lead_doc", "visibility", "original_proto_path", "position")
    DOCSTRING_FIELD_NUMBER: _ClassVar[int]
    LEAD_DOC_FIELD_NUMBER: _ClassVar[int]
    VISIBILITY_FIELD_NUMBER: _ClassVar[int]
    ORIGINAL_PROTO_PATH_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    docstring: str
    lead_doc: str
    visibility: Visibility
    original_proto_path: _containers.RepeatedScalarFieldContainer[str]
    position: int
    def __init__(self, docstring: _Optional[str] = ..., lead_doc: _Optional[str] = ..., visibility: _Optional[_Union[Visibility, str]] = ..., original_proto_path: _Optional[_Iterable[str]] = ..., position: _Optional[int] = ...) -> None: ...

import databricks_pb2 as _databricks_pb2
from scalapb import scalapb_pb2 as _scalapb_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import service as _service
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ArtifactCredentialType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    AZURE_SAS_URI: _ClassVar[ArtifactCredentialType]
    AWS_PRESIGNED_URL: _ClassVar[ArtifactCredentialType]
    GCP_SIGNED_URL: _ClassVar[ArtifactCredentialType]
    AZURE_ADLS_GEN2_SAS_URI: _ClassVar[ArtifactCredentialType]
AZURE_SAS_URI: ArtifactCredentialType
AWS_PRESIGNED_URL: ArtifactCredentialType
GCP_SIGNED_URL: ArtifactCredentialType
AZURE_ADLS_GEN2_SAS_URI: ArtifactCredentialType

class ArtifactCredentialInfo(_message.Message):
    __slots__ = ("run_id", "path", "signed_uri", "headers", "type")
    class HttpHeader(_message.Message):
        __slots__ = ("name", "value")
        NAME_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        name: str
        value: str
        def __init__(self, name: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    SIGNED_URI_FIELD_NUMBER: _ClassVar[int]
    HEADERS_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    path: str
    signed_uri: str
    headers: _containers.RepeatedCompositeFieldContainer[ArtifactCredentialInfo.HttpHeader]
    type: ArtifactCredentialType
    def __init__(self, run_id: _Optional[str] = ..., path: _Optional[str] = ..., signed_uri: _Optional[str] = ..., headers: _Optional[_Iterable[_Union[ArtifactCredentialInfo.HttpHeader, _Mapping]]] = ..., type: _Optional[_Union[ArtifactCredentialType, str]] = ...) -> None: ...

class LoggedModelArtifactCredential(_message.Message):
    __slots__ = ("model_id", "credential_info")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    CREDENTIAL_INFO_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    credential_info: ArtifactCredentialInfo
    def __init__(self, model_id: _Optional[str] = ..., credential_info: _Optional[_Union[ArtifactCredentialInfo, _Mapping]] = ...) -> None: ...

class GetCredentialsForRead(_message.Message):
    __slots__ = ("run_id", "path", "page_token")
    class Response(_message.Message):
        __slots__ = ("credential_infos", "next_page_token")
        CREDENTIAL_INFOS_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        credential_infos: _containers.RepeatedCompositeFieldContainer[ArtifactCredentialInfo]
        next_page_token: str
        def __init__(self, credential_infos: _Optional[_Iterable[_Union[ArtifactCredentialInfo, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    path: _containers.RepeatedScalarFieldContainer[str]
    page_token: str
    def __init__(self, run_id: _Optional[str] = ..., path: _Optional[_Iterable[str]] = ..., page_token: _Optional[str] = ...) -> None: ...

class GetCredentialsForWrite(_message.Message):
    __slots__ = ("run_id", "path", "page_token")
    class Response(_message.Message):
        __slots__ = ("credential_infos", "next_page_token")
        CREDENTIAL_INFOS_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        credential_infos: _containers.RepeatedCompositeFieldContainer[ArtifactCredentialInfo]
        next_page_token: str
        def __init__(self, credential_infos: _Optional[_Iterable[_Union[ArtifactCredentialInfo, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    path: _containers.RepeatedScalarFieldContainer[str]
    page_token: str
    def __init__(self, run_id: _Optional[str] = ..., path: _Optional[_Iterable[str]] = ..., page_token: _Optional[str] = ...) -> None: ...

class CreateMultipartUpload(_message.Message):
    __slots__ = ("run_id", "path", "num_parts")
    class Response(_message.Message):
        __slots__ = ("upload_id", "upload_credential_infos", "abort_credential_info")
        UPLOAD_ID_FIELD_NUMBER: _ClassVar[int]
        UPLOAD_CREDENTIAL_INFOS_FIELD_NUMBER: _ClassVar[int]
        ABORT_CREDENTIAL_INFO_FIELD_NUMBER: _ClassVar[int]
        upload_id: str
        upload_credential_infos: _containers.RepeatedCompositeFieldContainer[ArtifactCredentialInfo]
        abort_credential_info: ArtifactCredentialInfo
        def __init__(self, upload_id: _Optional[str] = ..., upload_credential_infos: _Optional[_Iterable[_Union[ArtifactCredentialInfo, _Mapping]]] = ..., abort_credential_info: _Optional[_Union[ArtifactCredentialInfo, _Mapping]] = ...) -> None: ...
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    NUM_PARTS_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    path: str
    num_parts: int
    def __init__(self, run_id: _Optional[str] = ..., path: _Optional[str] = ..., num_parts: _Optional[int] = ...) -> None: ...

class PartEtag(_message.Message):
    __slots__ = ("part_number", "etag")
    PART_NUMBER_FIELD_NUMBER: _ClassVar[int]
    ETAG_FIELD_NUMBER: _ClassVar[int]
    part_number: int
    etag: str
    def __init__(self, part_number: _Optional[int] = ..., etag: _Optional[str] = ...) -> None: ...

class CompleteMultipartUpload(_message.Message):
    __slots__ = ("run_id", "path", "upload_id", "part_etags")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    UPLOAD_ID_FIELD_NUMBER: _ClassVar[int]
    PART_ETAGS_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    path: str
    upload_id: str
    part_etags: _containers.RepeatedCompositeFieldContainer[PartEtag]
    def __init__(self, run_id: _Optional[str] = ..., path: _Optional[str] = ..., upload_id: _Optional[str] = ..., part_etags: _Optional[_Iterable[_Union[PartEtag, _Mapping]]] = ...) -> None: ...

class GetPresignedUploadPartUrl(_message.Message):
    __slots__ = ("run_id", "path", "upload_id", "part_number")
    class Response(_message.Message):
        __slots__ = ("upload_credential_info",)
        UPLOAD_CREDENTIAL_INFO_FIELD_NUMBER: _ClassVar[int]
        upload_credential_info: ArtifactCredentialInfo
        def __init__(self, upload_credential_info: _Optional[_Union[ArtifactCredentialInfo, _Mapping]] = ...) -> None: ...
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    UPLOAD_ID_FIELD_NUMBER: _ClassVar[int]
    PART_NUMBER_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    path: str
    upload_id: str
    part_number: int
    def __init__(self, run_id: _Optional[str] = ..., path: _Optional[str] = ..., upload_id: _Optional[str] = ..., part_number: _Optional[int] = ...) -> None: ...

class GetCredentialsForTraceDataDownload(_message.Message):
    __slots__ = ("request_id",)
    class Response(_message.Message):
        __slots__ = ("credential_info",)
        CREDENTIAL_INFO_FIELD_NUMBER: _ClassVar[int]
        credential_info: ArtifactCredentialInfo
        def __init__(self, credential_info: _Optional[_Union[ArtifactCredentialInfo, _Mapping]] = ...) -> None: ...
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    def __init__(self, request_id: _Optional[str] = ...) -> None: ...

class GetCredentialsForTraceDataUpload(_message.Message):
    __slots__ = ("request_id",)
    class Response(_message.Message):
        __slots__ = ("credential_info",)
        CREDENTIAL_INFO_FIELD_NUMBER: _ClassVar[int]
        credential_info: ArtifactCredentialInfo
        def __init__(self, credential_info: _Optional[_Union[ArtifactCredentialInfo, _Mapping]] = ...) -> None: ...
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    def __init__(self, request_id: _Optional[str] = ...) -> None: ...

class GetCredentialsForLoggedModelUpload(_message.Message):
    __slots__ = ("model_id", "paths", "page_token")
    class Response(_message.Message):
        __slots__ = ("credentials", "next_page_token")
        CREDENTIALS_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        credentials: _containers.RepeatedCompositeFieldContainer[LoggedModelArtifactCredential]
        next_page_token: str
        def __init__(self, credentials: _Optional[_Iterable[_Union[LoggedModelArtifactCredential, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    PATHS_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    paths: _containers.RepeatedScalarFieldContainer[str]
    page_token: str
    def __init__(self, model_id: _Optional[str] = ..., paths: _Optional[_Iterable[str]] = ..., page_token: _Optional[str] = ...) -> None: ...

class GetCredentialsForLoggedModelDownload(_message.Message):
    __slots__ = ("model_id", "paths", "page_token")
    class Response(_message.Message):
        __slots__ = ("credentials", "next_page_token")
        CREDENTIALS_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        credentials: _containers.RepeatedCompositeFieldContainer[LoggedModelArtifactCredential]
        next_page_token: str
        def __init__(self, credentials: _Optional[_Iterable[_Union[LoggedModelArtifactCredential, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    PATHS_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    paths: _containers.RepeatedScalarFieldContainer[str]
    page_token: str
    def __init__(self, model_id: _Optional[str] = ..., paths: _Optional[_Iterable[str]] = ..., page_token: _Optional[str] = ...) -> None: ...

class DatabricksMlflowArtifactsService(_service.service): ...

class DatabricksMlflowArtifactsService_Stub(DatabricksMlflowArtifactsService): ...

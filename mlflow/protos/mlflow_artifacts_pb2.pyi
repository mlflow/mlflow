import databricks_pb2 as _databricks_pb2
from scalapb import scalapb_pb2 as _scalapb_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import service as _service
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DownloadArtifact(_message.Message):
    __slots__ = ()
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    def __init__(self) -> None: ...

class UploadArtifact(_message.Message):
    __slots__ = ()
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    def __init__(self) -> None: ...

class ListArtifacts(_message.Message):
    __slots__ = ("path",)
    class Response(_message.Message):
        __slots__ = ("files",)
        FILES_FIELD_NUMBER: _ClassVar[int]
        files: _containers.RepeatedCompositeFieldContainer[FileInfo]
        def __init__(self, files: _Optional[_Iterable[_Union[FileInfo, _Mapping]]] = ...) -> None: ...
    PATH_FIELD_NUMBER: _ClassVar[int]
    path: str
    def __init__(self, path: _Optional[str] = ...) -> None: ...

class DeleteArtifact(_message.Message):
    __slots__ = ()
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    def __init__(self) -> None: ...

class FileInfo(_message.Message):
    __slots__ = ("path", "is_dir", "file_size")
    PATH_FIELD_NUMBER: _ClassVar[int]
    IS_DIR_FIELD_NUMBER: _ClassVar[int]
    FILE_SIZE_FIELD_NUMBER: _ClassVar[int]
    path: str
    is_dir: bool
    file_size: int
    def __init__(self, path: _Optional[str] = ..., is_dir: bool = ..., file_size: _Optional[int] = ...) -> None: ...

class CreateMultipartUpload(_message.Message):
    __slots__ = ("path", "num_parts")
    class Response(_message.Message):
        __slots__ = ("upload_id", "credentials")
        UPLOAD_ID_FIELD_NUMBER: _ClassVar[int]
        CREDENTIALS_FIELD_NUMBER: _ClassVar[int]
        upload_id: str
        credentials: _containers.RepeatedCompositeFieldContainer[MultipartUploadCredential]
        def __init__(self, upload_id: _Optional[str] = ..., credentials: _Optional[_Iterable[_Union[MultipartUploadCredential, _Mapping]]] = ...) -> None: ...
    PATH_FIELD_NUMBER: _ClassVar[int]
    NUM_PARTS_FIELD_NUMBER: _ClassVar[int]
    path: str
    num_parts: int
    def __init__(self, path: _Optional[str] = ..., num_parts: _Optional[int] = ...) -> None: ...

class CompleteMultipartUpload(_message.Message):
    __slots__ = ("path", "upload_id", "parts")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    PATH_FIELD_NUMBER: _ClassVar[int]
    UPLOAD_ID_FIELD_NUMBER: _ClassVar[int]
    PARTS_FIELD_NUMBER: _ClassVar[int]
    path: str
    upload_id: str
    parts: _containers.RepeatedCompositeFieldContainer[MultipartUploadPart]
    def __init__(self, path: _Optional[str] = ..., upload_id: _Optional[str] = ..., parts: _Optional[_Iterable[_Union[MultipartUploadPart, _Mapping]]] = ...) -> None: ...

class AbortMultipartUpload(_message.Message):
    __slots__ = ("path", "upload_id")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    PATH_FIELD_NUMBER: _ClassVar[int]
    UPLOAD_ID_FIELD_NUMBER: _ClassVar[int]
    path: str
    upload_id: str
    def __init__(self, path: _Optional[str] = ..., upload_id: _Optional[str] = ...) -> None: ...

class MultipartUploadCredential(_message.Message):
    __slots__ = ("url", "part_number", "headers")
    class HeadersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    URL_FIELD_NUMBER: _ClassVar[int]
    PART_NUMBER_FIELD_NUMBER: _ClassVar[int]
    HEADERS_FIELD_NUMBER: _ClassVar[int]
    url: str
    part_number: int
    headers: _containers.ScalarMap[str, str]
    def __init__(self, url: _Optional[str] = ..., part_number: _Optional[int] = ..., headers: _Optional[_Mapping[str, str]] = ...) -> None: ...

class MultipartUploadPart(_message.Message):
    __slots__ = ("part_number", "etag", "url")
    PART_NUMBER_FIELD_NUMBER: _ClassVar[int]
    ETAG_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    part_number: int
    etag: str
    url: str
    def __init__(self, part_number: _Optional[int] = ..., etag: _Optional[str] = ..., url: _Optional[str] = ...) -> None: ...

class MlflowArtifactsService(_service.service): ...

class MlflowArtifactsService_Stub(MlflowArtifactsService): ...

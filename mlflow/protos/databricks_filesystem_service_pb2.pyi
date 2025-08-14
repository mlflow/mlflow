import databricks_pb2 as _databricks_pb2
from scalapb import scalapb_pb2 as _scalapb_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import service as _service
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class HttpHeader(_message.Message):
    __slots__ = ("name", "value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: str
    def __init__(self, name: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class CreateDownloadUrlRequest(_message.Message):
    __slots__ = ("path",)
    PATH_FIELD_NUMBER: _ClassVar[int]
    path: str
    def __init__(self, path: _Optional[str] = ...) -> None: ...

class CreateDownloadUrlResponse(_message.Message):
    __slots__ = ("url", "headers")
    URL_FIELD_NUMBER: _ClassVar[int]
    HEADERS_FIELD_NUMBER: _ClassVar[int]
    url: str
    headers: _containers.RepeatedCompositeFieldContainer[HttpHeader]
    def __init__(self, url: _Optional[str] = ..., headers: _Optional[_Iterable[_Union[HttpHeader, _Mapping]]] = ...) -> None: ...

class CreateUploadUrlRequest(_message.Message):
    __slots__ = ("path",)
    PATH_FIELD_NUMBER: _ClassVar[int]
    path: str
    def __init__(self, path: _Optional[str] = ...) -> None: ...

class CreateUploadUrlResponse(_message.Message):
    __slots__ = ("url", "headers")
    URL_FIELD_NUMBER: _ClassVar[int]
    HEADERS_FIELD_NUMBER: _ClassVar[int]
    url: str
    headers: _containers.RepeatedCompositeFieldContainer[HttpHeader]
    def __init__(self, url: _Optional[str] = ..., headers: _Optional[_Iterable[_Union[HttpHeader, _Mapping]]] = ...) -> None: ...

class DirectoryEntry(_message.Message):
    __slots__ = ("path", "is_directory", "file_size", "last_modified", "name")
    PATH_FIELD_NUMBER: _ClassVar[int]
    IS_DIRECTORY_FIELD_NUMBER: _ClassVar[int]
    FILE_SIZE_FIELD_NUMBER: _ClassVar[int]
    LAST_MODIFIED_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    path: str
    is_directory: bool
    file_size: int
    last_modified: int
    name: str
    def __init__(self, path: _Optional[str] = ..., is_directory: bool = ..., file_size: _Optional[int] = ..., last_modified: _Optional[int] = ..., name: _Optional[str] = ...) -> None: ...

class ListDirectoryResponse(_message.Message):
    __slots__ = ("contents", "next_page_token")
    CONTENTS_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    contents: _containers.RepeatedCompositeFieldContainer[DirectoryEntry]
    next_page_token: str
    def __init__(self, contents: _Optional[_Iterable[_Union[DirectoryEntry, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...

class FilesystemService(_service.service): ...

class FilesystemService_Stub(FilesystemService): ...

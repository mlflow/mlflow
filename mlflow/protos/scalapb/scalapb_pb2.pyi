from google.protobuf import descriptor_pb2 as _descriptor_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor
OPTIONS_FIELD_NUMBER: _ClassVar[int]
options: _descriptor.FieldDescriptor
MESSAGE_FIELD_NUMBER: _ClassVar[int]
message: _descriptor.FieldDescriptor
FIELD_FIELD_NUMBER: _ClassVar[int]
field: _descriptor.FieldDescriptor

class ScalaPbOptions(_message.Message):
    __slots__ = ("package_name", "flat_package")
    PACKAGE_NAME_FIELD_NUMBER: _ClassVar[int]
    FLAT_PACKAGE_FIELD_NUMBER: _ClassVar[int]
    IMPORT_FIELD_NUMBER: _ClassVar[int]
    package_name: str
    flat_package: bool
    def __init__(self, package_name: _Optional[str] = ..., flat_package: bool = ..., **kwargs) -> None: ...

class MessageOptions(_message.Message):
    __slots__ = ("extends",)
    EXTENDS_FIELD_NUMBER: _ClassVar[int]
    extends: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, extends: _Optional[_Iterable[str]] = ...) -> None: ...

class FieldOptions(_message.Message):
    __slots__ = ("type",)
    TYPE_FIELD_NUMBER: _ClassVar[int]
    type: str
    def __init__(self, type: _Optional[str] = ...) -> None: ...

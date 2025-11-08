from scalapb import scalapb_pb2 as _scalapb_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class TableInfo(_message.Message):
    __slots__ = ("full_name", "table_id")
    FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    TABLE_ID_FIELD_NUMBER: _ClassVar[int]
    full_name: str
    table_id: str
    def __init__(self, full_name: _Optional[str] = ..., table_id: _Optional[str] = ...) -> None: ...

class GetTable(_message.Message):
    __slots__ = ("full_name_arg", "omit_columns", "omit_properties", "omit_constraints", "omit_dependencies", "omit_username", "omit_storage_credential_name")
    FULL_NAME_ARG_FIELD_NUMBER: _ClassVar[int]
    OMIT_COLUMNS_FIELD_NUMBER: _ClassVar[int]
    OMIT_PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    OMIT_CONSTRAINTS_FIELD_NUMBER: _ClassVar[int]
    OMIT_DEPENDENCIES_FIELD_NUMBER: _ClassVar[int]
    OMIT_USERNAME_FIELD_NUMBER: _ClassVar[int]
    OMIT_STORAGE_CREDENTIAL_NAME_FIELD_NUMBER: _ClassVar[int]
    full_name_arg: str
    omit_columns: bool
    omit_properties: bool
    omit_constraints: bool
    omit_dependencies: bool
    omit_username: bool
    omit_storage_credential_name: bool
    def __init__(self, full_name_arg: _Optional[str] = ..., omit_columns: bool = ..., omit_properties: bool = ..., omit_constraints: bool = ..., omit_dependencies: bool = ..., omit_username: bool = ..., omit_storage_credential_name: bool = ...) -> None: ...

class GetTableResponse(_message.Message):
    __slots__ = ("full_name", "table_id")
    FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    TABLE_ID_FIELD_NUMBER: _ClassVar[int]
    full_name: str
    table_id: str
    def __init__(self, full_name: _Optional[str] = ..., table_id: _Optional[str] = ...) -> None: ...

from scalapb import scalapb_pb2 as _scalapb_pb2
import databricks_pb2 as _databricks_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import service as _service
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SecretScope(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    GLOBAL: _ClassVar[SecretScope]
    SCORER: _ClassVar[SecretScope]
GLOBAL: SecretScope
SCORER: SecretScope

class CreateSecret(_message.Message):
    __slots__ = ("encrypted_name", "encrypted_value", "encrypted_dek", "scope", "scope_id", "integrity_hash")
    class Response(_message.Message):
        __slots__ = ("success",)
        SUCCESS_FIELD_NUMBER: _ClassVar[int]
        success: bool
        def __init__(self, success: bool = ...) -> None: ...
    ENCRYPTED_NAME_FIELD_NUMBER: _ClassVar[int]
    ENCRYPTED_VALUE_FIELD_NUMBER: _ClassVar[int]
    ENCRYPTED_DEK_FIELD_NUMBER: _ClassVar[int]
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    SCOPE_ID_FIELD_NUMBER: _ClassVar[int]
    INTEGRITY_HASH_FIELD_NUMBER: _ClassVar[int]
    encrypted_name: str
    encrypted_value: str
    encrypted_dek: str
    scope: SecretScope
    scope_id: int
    integrity_hash: str
    def __init__(self, encrypted_name: _Optional[str] = ..., encrypted_value: _Optional[str] = ..., encrypted_dek: _Optional[str] = ..., scope: _Optional[_Union[SecretScope, str]] = ..., scope_id: _Optional[int] = ..., integrity_hash: _Optional[str] = ...) -> None: ...

class ListSecrets(_message.Message):
    __slots__ = ("scope", "scope_id")
    class Response(_message.Message):
        __slots__ = ("secret_names",)
        SECRET_NAMES_FIELD_NUMBER: _ClassVar[int]
        secret_names: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, secret_names: _Optional[_Iterable[str]] = ...) -> None: ...
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    SCOPE_ID_FIELD_NUMBER: _ClassVar[int]
    scope: SecretScope
    scope_id: int
    def __init__(self, scope: _Optional[_Union[SecretScope, str]] = ..., scope_id: _Optional[int] = ...) -> None: ...

class DeleteSecret(_message.Message):
    __slots__ = ("name", "scope", "scope_id")
    class Response(_message.Message):
        __slots__ = ("success",)
        SUCCESS_FIELD_NUMBER: _ClassVar[int]
        success: bool
        def __init__(self, success: bool = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    SCOPE_ID_FIELD_NUMBER: _ClassVar[int]
    name: str
    scope: SecretScope
    scope_id: int
    def __init__(self, name: _Optional[str] = ..., scope: _Optional[_Union[SecretScope, str]] = ..., scope_id: _Optional[int] = ...) -> None: ...

class SecretsService(_service.service): ...

class SecretsService_Stub(SecretsService): ...

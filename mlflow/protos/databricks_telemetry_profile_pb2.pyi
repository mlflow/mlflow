import databricks_pb2 as _databricks_pb2
from scalapb import scalapb_pb2 as _scalapb_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import service as _service
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GetTelemetryProfileRequest(_message.Message):
    __slots__ = ("profile_id",)
    PROFILE_ID_FIELD_NUMBER: _ClassVar[int]
    profile_id: str
    def __init__(self, profile_id: _Optional[str] = ...) -> None: ...

class TelemetryProfile(_message.Message):
    __slots__ = ("profile_id", "profile_name", "created_at", "created_by", "updated_at", "updated_by", "exporters")
    PROFILE_ID_FIELD_NUMBER: _ClassVar[int]
    PROFILE_NAME_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    CREATED_BY_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_BY_FIELD_NUMBER: _ClassVar[int]
    EXPORTERS_FIELD_NUMBER: _ClassVar[int]
    profile_id: str
    profile_name: str
    created_at: int
    created_by: str
    updated_at: int
    updated_by: str
    exporters: _containers.RepeatedCompositeFieldContainer[Exporter]
    def __init__(self, profile_id: _Optional[str] = ..., profile_name: _Optional[str] = ..., created_at: _Optional[int] = ..., created_by: _Optional[str] = ..., updated_at: _Optional[int] = ..., updated_by: _Optional[str] = ..., exporters: _Optional[_Iterable[_Union[Exporter, _Mapping]]] = ...) -> None: ...

class Exporter(_message.Message):
    __slots__ = ("type", "uc_tables")
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        TYPE_UNSPECIFIED: _ClassVar[Exporter.Type]
        UNITY_CATALOG_TABLES: _ClassVar[Exporter.Type]
    TYPE_UNSPECIFIED: Exporter.Type
    UNITY_CATALOG_TABLES: Exporter.Type
    TYPE_FIELD_NUMBER: _ClassVar[int]
    UC_TABLES_FIELD_NUMBER: _ClassVar[int]
    type: Exporter.Type
    uc_tables: UnityCatalogTablesConfig
    def __init__(self, type: _Optional[_Union[Exporter.Type, str]] = ..., uc_tables: _Optional[_Union[UnityCatalogTablesConfig, _Mapping]] = ...) -> None: ...

class UnityCatalogTablesConfig(_message.Message):
    __slots__ = ("uc_catalog", "uc_schema", "uc_table_prefix", "logs_table_name", "logs_schema_version", "metrics_table_name", "metrics_schema_version", "spans_table_name", "spans_schema_version")
    UC_CATALOG_FIELD_NUMBER: _ClassVar[int]
    UC_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    UC_TABLE_PREFIX_FIELD_NUMBER: _ClassVar[int]
    LOGS_TABLE_NAME_FIELD_NUMBER: _ClassVar[int]
    LOGS_SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    METRICS_TABLE_NAME_FIELD_NUMBER: _ClassVar[int]
    METRICS_SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    SPANS_TABLE_NAME_FIELD_NUMBER: _ClassVar[int]
    SPANS_SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    uc_catalog: str
    uc_schema: str
    uc_table_prefix: str
    logs_table_name: str
    logs_schema_version: str
    metrics_table_name: str
    metrics_schema_version: str
    spans_table_name: str
    spans_schema_version: str
    def __init__(self, uc_catalog: _Optional[str] = ..., uc_schema: _Optional[str] = ..., uc_table_prefix: _Optional[str] = ..., logs_table_name: _Optional[str] = ..., logs_schema_version: _Optional[str] = ..., metrics_table_name: _Optional[str] = ..., metrics_schema_version: _Optional[str] = ..., spans_table_name: _Optional[str] = ..., spans_schema_version: _Optional[str] = ...) -> None: ...

class DatabricksTracingServerService(_service.service): ...

class DatabricksTracingServerService_Stub(DatabricksTracingServerService): ...

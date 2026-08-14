from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Dataset(_message.Message):
    __slots__ = ("dataset_id", "name", "tags", "schema", "profile", "digest", "created_time", "last_update_time", "created_by", "last_updated_by", "experiment_ids")
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    PROFILE_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    CREATED_TIME_FIELD_NUMBER: _ClassVar[int]
    LAST_UPDATE_TIME_FIELD_NUMBER: _ClassVar[int]
    CREATED_BY_FIELD_NUMBER: _ClassVar[int]
    LAST_UPDATED_BY_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_IDS_FIELD_NUMBER: _ClassVar[int]
    dataset_id: str
    name: str
    tags: str
    schema: str
    profile: str
    digest: str
    created_time: int
    last_update_time: int
    created_by: str
    last_updated_by: str
    experiment_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, dataset_id: _Optional[str] = ..., name: _Optional[str] = ..., tags: _Optional[str] = ..., schema: _Optional[str] = ..., profile: _Optional[str] = ..., digest: _Optional[str] = ..., created_time: _Optional[int] = ..., last_update_time: _Optional[int] = ..., created_by: _Optional[str] = ..., last_updated_by: _Optional[str] = ..., experiment_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class DatasetRecord(_message.Message):
    __slots__ = ("dataset_record_id", "dataset_id", "inputs", "expectations", "tags", "source", "source_id", "source_type", "created_time", "last_update_time", "created_by", "last_updated_by", "outputs")
    DATASET_RECORD_ID_FIELD_NUMBER: _ClassVar[int]
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    EXPECTATIONS_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    CREATED_TIME_FIELD_NUMBER: _ClassVar[int]
    LAST_UPDATE_TIME_FIELD_NUMBER: _ClassVar[int]
    CREATED_BY_FIELD_NUMBER: _ClassVar[int]
    LAST_UPDATED_BY_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    dataset_record_id: str
    dataset_id: str
    inputs: str
    expectations: str
    tags: str
    source: str
    source_id: str
    source_type: DatasetRecordSource.SourceType
    created_time: int
    last_update_time: int
    created_by: str
    last_updated_by: str
    outputs: str
    def __init__(self, dataset_record_id: _Optional[str] = ..., dataset_id: _Optional[str] = ..., inputs: _Optional[str] = ..., expectations: _Optional[str] = ..., tags: _Optional[str] = ..., source: _Optional[str] = ..., source_id: _Optional[str] = ..., source_type: _Optional[_Union[DatasetRecordSource.SourceType, str]] = ..., created_time: _Optional[int] = ..., last_update_time: _Optional[int] = ..., created_by: _Optional[str] = ..., last_updated_by: _Optional[str] = ..., outputs: _Optional[str] = ...) -> None: ...

class DatasetRecordSource(_message.Message):
    __slots__ = ("source_type", "source_data")
    class SourceType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        SOURCE_TYPE_UNSPECIFIED: _ClassVar[DatasetRecordSource.SourceType]
        TRACE: _ClassVar[DatasetRecordSource.SourceType]
        HUMAN: _ClassVar[DatasetRecordSource.SourceType]
        DOCUMENT: _ClassVar[DatasetRecordSource.SourceType]
        CODE: _ClassVar[DatasetRecordSource.SourceType]
    SOURCE_TYPE_UNSPECIFIED: DatasetRecordSource.SourceType
    TRACE: DatasetRecordSource.SourceType
    HUMAN: DatasetRecordSource.SourceType
    DOCUMENT: DatasetRecordSource.SourceType
    CODE: DatasetRecordSource.SourceType
    SOURCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_DATA_FIELD_NUMBER: _ClassVar[int]
    source_type: DatasetRecordSource.SourceType
    source_data: str
    def __init__(self, source_type: _Optional[_Union[DatasetRecordSource.SourceType, str]] = ..., source_data: _Optional[str] = ...) -> None: ...

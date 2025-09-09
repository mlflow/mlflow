import databricks_pb2 as _databricks_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AssessmentSource(_message.Message):
    __slots__ = ("source_type", "source_id")
    class SourceType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        SOURCE_TYPE_UNSPECIFIED: _ClassVar[AssessmentSource.SourceType]
        HUMAN: _ClassVar[AssessmentSource.SourceType]
        LLM_JUDGE: _ClassVar[AssessmentSource.SourceType]
        CODE: _ClassVar[AssessmentSource.SourceType]
    SOURCE_TYPE_UNSPECIFIED: AssessmentSource.SourceType
    HUMAN: AssessmentSource.SourceType
    LLM_JUDGE: AssessmentSource.SourceType
    CODE: AssessmentSource.SourceType
    SOURCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    source_type: AssessmentSource.SourceType
    source_id: str
    def __init__(self, source_type: _Optional[_Union[AssessmentSource.SourceType, str]] = ..., source_id: _Optional[str] = ...) -> None: ...

class AssessmentError(_message.Message):
    __slots__ = ("error_code", "error_message", "stack_trace")
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    STACK_TRACE_FIELD_NUMBER: _ClassVar[int]
    error_code: str
    error_message: str
    stack_trace: str
    def __init__(self, error_code: _Optional[str] = ..., error_message: _Optional[str] = ..., stack_trace: _Optional[str] = ...) -> None: ...

class Expectation(_message.Message):
    __slots__ = ("value", "serialized_value")
    class SerializedValue(_message.Message):
        __slots__ = ("serialization_format", "value")
        SERIALIZATION_FORMAT_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        serialization_format: str
        value: str
        def __init__(self, serialization_format: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    VALUE_FIELD_NUMBER: _ClassVar[int]
    SERIALIZED_VALUE_FIELD_NUMBER: _ClassVar[int]
    value: _struct_pb2.Value
    serialized_value: Expectation.SerializedValue
    def __init__(self, value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ..., serialized_value: _Optional[_Union[Expectation.SerializedValue, _Mapping]] = ...) -> None: ...

class Feedback(_message.Message):
    __slots__ = ("value", "error")
    VALUE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    value: _struct_pb2.Value
    error: AssessmentError
    def __init__(self, value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ..., error: _Optional[_Union[AssessmentError, _Mapping]] = ...) -> None: ...

class Assessment(_message.Message):
    __slots__ = ("assessment_id", "assessment_name", "trace_id", "span_id", "source", "create_time", "last_update_time", "feedback", "expectation", "rationale", "error", "metadata", "overrides", "valid")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ASSESSMENT_ID_FIELD_NUMBER: _ClassVar[int]
    ASSESSMENT_NAME_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    SPAN_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    CREATE_TIME_FIELD_NUMBER: _ClassVar[int]
    LAST_UPDATE_TIME_FIELD_NUMBER: _ClassVar[int]
    FEEDBACK_FIELD_NUMBER: _ClassVar[int]
    EXPECTATION_FIELD_NUMBER: _ClassVar[int]
    RATIONALE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    OVERRIDES_FIELD_NUMBER: _ClassVar[int]
    VALID_FIELD_NUMBER: _ClassVar[int]
    assessment_id: str
    assessment_name: str
    trace_id: str
    span_id: str
    source: AssessmentSource
    create_time: _timestamp_pb2.Timestamp
    last_update_time: _timestamp_pb2.Timestamp
    feedback: Feedback
    expectation: Expectation
    rationale: str
    error: AssessmentError
    metadata: _containers.ScalarMap[str, str]
    overrides: str
    valid: bool
    def __init__(self, assessment_id: _Optional[str] = ..., assessment_name: _Optional[str] = ..., trace_id: _Optional[str] = ..., span_id: _Optional[str] = ..., source: _Optional[_Union[AssessmentSource, _Mapping]] = ..., create_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., last_update_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., feedback: _Optional[_Union[Feedback, _Mapping]] = ..., expectation: _Optional[_Union[Expectation, _Mapping]] = ..., rationale: _Optional[str] = ..., error: _Optional[_Union[AssessmentError, _Mapping]] = ..., metadata: _Optional[_Mapping[str, str]] = ..., overrides: _Optional[str] = ..., valid: bool = ...) -> None: ...

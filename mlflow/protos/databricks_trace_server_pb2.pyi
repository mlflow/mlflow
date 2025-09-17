import databricks_pb2 as _databricks_pb2
import assessments_pb2 as _assessments_pb2
from google.protobuf import duration_pb2 as _duration_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from scalapb import scalapb_pb2 as _scalapb_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import service as _service
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CreateTrace(_message.Message):
    __slots__ = ("info", "data")
    INFO_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    info: TraceInfo
    data: TraceData
    def __init__(self, info: _Optional[_Union[TraceInfo, _Mapping]] = ..., data: _Optional[_Union[TraceData, _Mapping]] = ...) -> None: ...

class Trace(_message.Message):
    __slots__ = ("info", "data")
    INFO_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    info: TraceInfo
    data: TraceData
    def __init__(self, info: _Optional[_Union[TraceInfo, _Mapping]] = ..., data: _Optional[_Union[TraceData, _Mapping]] = ...) -> None: ...

class TraceInfo(_message.Message):
    __slots__ = ("trace_id", "client_request_id", "trace_location", "request", "response", "request_preview", "response_preview", "request_time", "execution_duration", "state", "trace_metadata", "assessments", "tags")
    class State(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STATE_UNSPECIFIED: _ClassVar[TraceInfo.State]
        OK: _ClassVar[TraceInfo.State]
        ERROR: _ClassVar[TraceInfo.State]
        IN_PROGRESS: _ClassVar[TraceInfo.State]
    STATE_UNSPECIFIED: TraceInfo.State
    OK: TraceInfo.State
    ERROR: TraceInfo.State
    IN_PROGRESS: TraceInfo.State
    class TraceMetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class TagsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    CLIENT_REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TRACE_LOCATION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_PREVIEW_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_PREVIEW_FIELD_NUMBER: _ClassVar[int]
    REQUEST_TIME_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_DURATION_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    TRACE_METADATA_FIELD_NUMBER: _ClassVar[int]
    ASSESSMENTS_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    trace_id: str
    client_request_id: str
    trace_location: TraceLocation
    request: str
    response: str
    request_preview: str
    response_preview: str
    request_time: _timestamp_pb2.Timestamp
    execution_duration: _duration_pb2.Duration
    state: TraceInfo.State
    trace_metadata: _containers.ScalarMap[str, str]
    assessments: _containers.RepeatedCompositeFieldContainer[_assessments_pb2.Assessment]
    tags: _containers.ScalarMap[str, str]
    def __init__(self, trace_id: _Optional[str] = ..., client_request_id: _Optional[str] = ..., trace_location: _Optional[_Union[TraceLocation, _Mapping]] = ..., request: _Optional[str] = ..., response: _Optional[str] = ..., request_preview: _Optional[str] = ..., response_preview: _Optional[str] = ..., request_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., execution_duration: _Optional[_Union[_duration_pb2.Duration, _Mapping]] = ..., state: _Optional[_Union[TraceInfo.State, str]] = ..., trace_metadata: _Optional[_Mapping[str, str]] = ..., assessments: _Optional[_Iterable[_Union[_assessments_pb2.Assessment, _Mapping]]] = ..., tags: _Optional[_Mapping[str, str]] = ...) -> None: ...

class TraceLocation(_message.Message):
    __slots__ = ("type", "mlflow_experiment", "inference_table")
    class TraceLocationType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        TRACE_LOCATION_TYPE_UNSPECIFIED: _ClassVar[TraceLocation.TraceLocationType]
        MLFLOW_EXPERIMENT: _ClassVar[TraceLocation.TraceLocationType]
        INFERENCE_TABLE: _ClassVar[TraceLocation.TraceLocationType]
    TRACE_LOCATION_TYPE_UNSPECIFIED: TraceLocation.TraceLocationType
    MLFLOW_EXPERIMENT: TraceLocation.TraceLocationType
    INFERENCE_TABLE: TraceLocation.TraceLocationType
    class MlflowExperimentLocation(_message.Message):
        __slots__ = ("experiment_id",)
        EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
        experiment_id: str
        def __init__(self, experiment_id: _Optional[str] = ...) -> None: ...
    class InferenceTableLocation(_message.Message):
        __slots__ = ("full_table_name",)
        FULL_TABLE_NAME_FIELD_NUMBER: _ClassVar[int]
        full_table_name: str
        def __init__(self, full_table_name: _Optional[str] = ...) -> None: ...
    TYPE_FIELD_NUMBER: _ClassVar[int]
    MLFLOW_EXPERIMENT_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_TABLE_FIELD_NUMBER: _ClassVar[int]
    type: TraceLocation.TraceLocationType
    mlflow_experiment: TraceLocation.MlflowExperimentLocation
    inference_table: TraceLocation.InferenceTableLocation
    def __init__(self, type: _Optional[_Union[TraceLocation.TraceLocationType, str]] = ..., mlflow_experiment: _Optional[_Union[TraceLocation.MlflowExperimentLocation, _Mapping]] = ..., inference_table: _Optional[_Union[TraceLocation.InferenceTableLocation, _Mapping]] = ...) -> None: ...

class TraceData(_message.Message):
    __slots__ = ("spans",)
    SPANS_FIELD_NUMBER: _ClassVar[int]
    spans: _containers.RepeatedCompositeFieldContainer[Span]
    def __init__(self, spans: _Optional[_Iterable[_Union[Span, _Mapping]]] = ...) -> None: ...

class Span(_message.Message):
    __slots__ = ("trace_id", "span_id", "trace_state", "parent_span_id", "flags", "name", "kind", "start_time_unix_nano", "end_time_unix_nano", "attributes", "dropped_attributes_count", "events", "dropped_events_count", "links", "dropped_links_count", "status")
    class SpanKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        SPAN_KIND_UNSPECIFIED: _ClassVar[Span.SpanKind]
        SPAN_KIND_INTERNAL: _ClassVar[Span.SpanKind]
        SPAN_KIND_SERVER: _ClassVar[Span.SpanKind]
        SPAN_KIND_CLIENT: _ClassVar[Span.SpanKind]
        SPAN_KIND_PRODUCER: _ClassVar[Span.SpanKind]
        SPAN_KIND_CONSUMER: _ClassVar[Span.SpanKind]
    SPAN_KIND_UNSPECIFIED: Span.SpanKind
    SPAN_KIND_INTERNAL: Span.SpanKind
    SPAN_KIND_SERVER: Span.SpanKind
    SPAN_KIND_CLIENT: Span.SpanKind
    SPAN_KIND_PRODUCER: Span.SpanKind
    SPAN_KIND_CONSUMER: Span.SpanKind
    class AttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    class Event(_message.Message):
        __slots__ = ("time_unix_nano", "name", "attributes", "dropped_attributes_count")
        class AttributesEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: _struct_pb2.Value
            def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
        TIME_UNIX_NANO_FIELD_NUMBER: _ClassVar[int]
        NAME_FIELD_NUMBER: _ClassVar[int]
        ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
        DROPPED_ATTRIBUTES_COUNT_FIELD_NUMBER: _ClassVar[int]
        time_unix_nano: int
        name: str
        attributes: _containers.MessageMap[str, _struct_pb2.Value]
        dropped_attributes_count: int
        def __init__(self, time_unix_nano: _Optional[int] = ..., name: _Optional[str] = ..., attributes: _Optional[_Mapping[str, _struct_pb2.Value]] = ..., dropped_attributes_count: _Optional[int] = ...) -> None: ...
    class Link(_message.Message):
        __slots__ = ("trace_id", "span_id", "trace_state", "attributes", "dropped_attributes_count", "flags")
        class AttributesEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: _struct_pb2.Value
            def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
        TRACE_ID_FIELD_NUMBER: _ClassVar[int]
        SPAN_ID_FIELD_NUMBER: _ClassVar[int]
        TRACE_STATE_FIELD_NUMBER: _ClassVar[int]
        ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
        DROPPED_ATTRIBUTES_COUNT_FIELD_NUMBER: _ClassVar[int]
        FLAGS_FIELD_NUMBER: _ClassVar[int]
        trace_id: bytes
        span_id: bytes
        trace_state: str
        attributes: _containers.MessageMap[str, _struct_pb2.Value]
        dropped_attributes_count: int
        flags: int
        def __init__(self, trace_id: _Optional[bytes] = ..., span_id: _Optional[bytes] = ..., trace_state: _Optional[str] = ..., attributes: _Optional[_Mapping[str, _struct_pb2.Value]] = ..., dropped_attributes_count: _Optional[int] = ..., flags: _Optional[int] = ...) -> None: ...
    class Status(_message.Message):
        __slots__ = ("message", "code")
        class StatusCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            STATUS_CODE_UNSET: _ClassVar[Span.Status.StatusCode]
            STATUS_CODE_OK: _ClassVar[Span.Status.StatusCode]
            STATUS_CODE_ERROR: _ClassVar[Span.Status.StatusCode]
        STATUS_CODE_UNSET: Span.Status.StatusCode
        STATUS_CODE_OK: Span.Status.StatusCode
        STATUS_CODE_ERROR: Span.Status.StatusCode
        MESSAGE_FIELD_NUMBER: _ClassVar[int]
        CODE_FIELD_NUMBER: _ClassVar[int]
        message: str
        code: Span.Status.StatusCode
        def __init__(self, message: _Optional[str] = ..., code: _Optional[_Union[Span.Status.StatusCode, str]] = ...) -> None: ...
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    SPAN_ID_FIELD_NUMBER: _ClassVar[int]
    TRACE_STATE_FIELD_NUMBER: _ClassVar[int]
    PARENT_SPAN_ID_FIELD_NUMBER: _ClassVar[int]
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    START_TIME_UNIX_NANO_FIELD_NUMBER: _ClassVar[int]
    END_TIME_UNIX_NANO_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    DROPPED_ATTRIBUTES_COUNT_FIELD_NUMBER: _ClassVar[int]
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    DROPPED_EVENTS_COUNT_FIELD_NUMBER: _ClassVar[int]
    LINKS_FIELD_NUMBER: _ClassVar[int]
    DROPPED_LINKS_COUNT_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    trace_id: bytes
    span_id: bytes
    trace_state: str
    parent_span_id: bytes
    flags: int
    name: str
    kind: Span.SpanKind
    start_time_unix_nano: int
    end_time_unix_nano: int
    attributes: _containers.MessageMap[str, _struct_pb2.Value]
    dropped_attributes_count: int
    events: _containers.RepeatedCompositeFieldContainer[Span.Event]
    dropped_events_count: int
    links: _containers.RepeatedCompositeFieldContainer[Span.Link]
    dropped_links_count: int
    status: Span.Status
    def __init__(self, trace_id: _Optional[bytes] = ..., span_id: _Optional[bytes] = ..., trace_state: _Optional[str] = ..., parent_span_id: _Optional[bytes] = ..., flags: _Optional[int] = ..., name: _Optional[str] = ..., kind: _Optional[_Union[Span.SpanKind, str]] = ..., start_time_unix_nano: _Optional[int] = ..., end_time_unix_nano: _Optional[int] = ..., attributes: _Optional[_Mapping[str, _struct_pb2.Value]] = ..., dropped_attributes_count: _Optional[int] = ..., events: _Optional[_Iterable[_Union[Span.Event, _Mapping]]] = ..., dropped_events_count: _Optional[int] = ..., links: _Optional[_Iterable[_Union[Span.Link, _Mapping]]] = ..., dropped_links_count: _Optional[int] = ..., status: _Optional[_Union[Span.Status, _Mapping]] = ...) -> None: ...

class DatabricksTracingServerService(_service.service): ...

class DatabricksTracingServerService_Stub(DatabricksTracingServerService): ...

import assessments_pb2 as _assessments_pb2
import databricks_pb2 as _databricks_pb2
from google.protobuf import duration_pb2 as _duration_pb2
from google.protobuf import field_mask_pb2 as _field_mask_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from opentelemetry.proto.trace.v1 import trace_pb2 as _trace_pb2
from scalapb import scalapb_pb2 as _scalapb_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import service as _service
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class UCSchemaLocation(_message.Message):
    __slots__ = ("catalog_name", "schema_name", "otel_spans_table_name", "otel_logs_table_name")
    CATALOG_NAME_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_NAME_FIELD_NUMBER: _ClassVar[int]
    OTEL_SPANS_TABLE_NAME_FIELD_NUMBER: _ClassVar[int]
    OTEL_LOGS_TABLE_NAME_FIELD_NUMBER: _ClassVar[int]
    catalog_name: str
    schema_name: str
    otel_spans_table_name: str
    otel_logs_table_name: str
    def __init__(self, catalog_name: _Optional[str] = ..., schema_name: _Optional[str] = ..., otel_spans_table_name: _Optional[str] = ..., otel_logs_table_name: _Optional[str] = ...) -> None: ...

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

class TraceLocation(_message.Message):
    __slots__ = ("type", "mlflow_experiment", "inference_table", "uc_schema")
    class TraceLocationType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        TRACE_LOCATION_TYPE_UNSPECIFIED: _ClassVar[TraceLocation.TraceLocationType]
        MLFLOW_EXPERIMENT: _ClassVar[TraceLocation.TraceLocationType]
        INFERENCE_TABLE: _ClassVar[TraceLocation.TraceLocationType]
        UC_SCHEMA: _ClassVar[TraceLocation.TraceLocationType]
    TRACE_LOCATION_TYPE_UNSPECIFIED: TraceLocation.TraceLocationType
    MLFLOW_EXPERIMENT: TraceLocation.TraceLocationType
    INFERENCE_TABLE: TraceLocation.TraceLocationType
    UC_SCHEMA: TraceLocation.TraceLocationType
    TYPE_FIELD_NUMBER: _ClassVar[int]
    MLFLOW_EXPERIMENT_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_TABLE_FIELD_NUMBER: _ClassVar[int]
    UC_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    type: TraceLocation.TraceLocationType
    mlflow_experiment: MlflowExperimentLocation
    inference_table: InferenceTableLocation
    uc_schema: UCSchemaLocation
    def __init__(self, type: _Optional[_Union[TraceLocation.TraceLocationType, str]] = ..., mlflow_experiment: _Optional[_Union[MlflowExperimentLocation, _Mapping]] = ..., inference_table: _Optional[_Union[InferenceTableLocation, _Mapping]] = ..., uc_schema: _Optional[_Union[UCSchemaLocation, _Mapping]] = ...) -> None: ...

class TraceInfo(_message.Message):
    __slots__ = ("trace_id", "client_request_id", "trace_location", "request_preview", "response_preview", "request_time", "execution_duration", "state", "trace_metadata", "assessments", "tags")
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
    request_preview: str
    response_preview: str
    request_time: _timestamp_pb2.Timestamp
    execution_duration: _duration_pb2.Duration
    state: TraceInfo.State
    trace_metadata: _containers.ScalarMap[str, str]
    assessments: _containers.RepeatedCompositeFieldContainer[Assessment]
    tags: _containers.ScalarMap[str, str]
    def __init__(self, trace_id: _Optional[str] = ..., client_request_id: _Optional[str] = ..., trace_location: _Optional[_Union[TraceLocation, _Mapping]] = ..., request_preview: _Optional[str] = ..., response_preview: _Optional[str] = ..., request_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., execution_duration: _Optional[_Union[_duration_pb2.Duration, _Mapping]] = ..., state: _Optional[_Union[TraceInfo.State, str]] = ..., trace_metadata: _Optional[_Mapping[str, str]] = ..., assessments: _Optional[_Iterable[_Union[Assessment, _Mapping]]] = ..., tags: _Optional[_Mapping[str, str]] = ...) -> None: ...

class CreateTraceInfo(_message.Message):
    __slots__ = ("location_id", "trace_info")
    class Response(_message.Message):
        __slots__ = ("trace_info",)
        TRACE_INFO_FIELD_NUMBER: _ClassVar[int]
        trace_info: TraceInfo
        def __init__(self, trace_info: _Optional[_Union[TraceInfo, _Mapping]] = ...) -> None: ...
    LOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    TRACE_INFO_FIELD_NUMBER: _ClassVar[int]
    location_id: str
    trace_info: TraceInfo
    def __init__(self, location_id: _Optional[str] = ..., trace_info: _Optional[_Union[TraceInfo, _Mapping]] = ...) -> None: ...

class TracePath(_message.Message):
    __slots__ = ("trace_location", "trace_id")
    TRACE_LOCATION_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    trace_location: TraceLocation
    trace_id: str
    def __init__(self, trace_location: _Optional[_Union[TraceLocation, _Mapping]] = ..., trace_id: _Optional[str] = ...) -> None: ...

class Trace(_message.Message):
    __slots__ = ("trace_info", "spans")
    TRACE_INFO_FIELD_NUMBER: _ClassVar[int]
    SPANS_FIELD_NUMBER: _ClassVar[int]
    trace_info: TraceInfo
    spans: _containers.RepeatedCompositeFieldContainer[_trace_pb2.Span]
    def __init__(self, trace_info: _Optional[_Union[TraceInfo, _Mapping]] = ..., spans: _Optional[_Iterable[_Union[_trace_pb2.Span, _Mapping]]] = ...) -> None: ...

class BatchGetTraces(_message.Message):
    __slots__ = ("location_id", "trace_ids", "sql_warehouse_id")
    class Response(_message.Message):
        __slots__ = ("traces",)
        TRACES_FIELD_NUMBER: _ClassVar[int]
        traces: _containers.RepeatedCompositeFieldContainer[Trace]
        def __init__(self, traces: _Optional[_Iterable[_Union[Trace, _Mapping]]] = ...) -> None: ...
    LOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    TRACE_IDS_FIELD_NUMBER: _ClassVar[int]
    SQL_WAREHOUSE_ID_FIELD_NUMBER: _ClassVar[int]
    location_id: str
    trace_ids: _containers.RepeatedScalarFieldContainer[str]
    sql_warehouse_id: str
    def __init__(self, location_id: _Optional[str] = ..., trace_ids: _Optional[_Iterable[str]] = ..., sql_warehouse_id: _Optional[str] = ...) -> None: ...

class GetTraceInfo(_message.Message):
    __slots__ = ("trace_id", "location", "sql_warehouse_id")
    class Response(_message.Message):
        __slots__ = ("trace",)
        TRACE_FIELD_NUMBER: _ClassVar[int]
        trace: Trace
        def __init__(self, trace: _Optional[_Union[Trace, _Mapping]] = ...) -> None: ...
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    SQL_WAREHOUSE_ID_FIELD_NUMBER: _ClassVar[int]
    trace_id: str
    location: str
    sql_warehouse_id: str
    def __init__(self, trace_id: _Optional[str] = ..., location: _Optional[str] = ..., sql_warehouse_id: _Optional[str] = ...) -> None: ...

class SetTraceTag(_message.Message):
    __slots__ = ("trace_id", "location_id", "key", "value")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    LOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    trace_id: str
    location_id: str
    key: str
    value: str
    def __init__(self, trace_id: _Optional[str] = ..., location_id: _Optional[str] = ..., key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class DeleteTraceTag(_message.Message):
    __slots__ = ("trace_id", "location_id", "key", "sql_warehouse_id")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    LOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    SQL_WAREHOUSE_ID_FIELD_NUMBER: _ClassVar[int]
    trace_id: str
    location_id: str
    key: str
    sql_warehouse_id: str
    def __init__(self, trace_id: _Optional[str] = ..., location_id: _Optional[str] = ..., key: _Optional[str] = ..., sql_warehouse_id: _Optional[str] = ...) -> None: ...

class SearchTraces(_message.Message):
    __slots__ = ("locations", "filter", "max_results", "order_by", "sql_warehouse_id", "page_token")
    class Response(_message.Message):
        __slots__ = ("trace_infos", "next_page_token")
        TRACE_INFOS_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        trace_infos: _containers.RepeatedCompositeFieldContainer[TraceInfo]
        next_page_token: str
        def __init__(self, trace_infos: _Optional[_Iterable[_Union[TraceInfo, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    LOCATIONS_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    ORDER_BY_FIELD_NUMBER: _ClassVar[int]
    SQL_WAREHOUSE_ID_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    locations: _containers.RepeatedCompositeFieldContainer[TraceLocation]
    filter: str
    max_results: int
    order_by: _containers.RepeatedScalarFieldContainer[str]
    sql_warehouse_id: str
    page_token: str
    def __init__(self, locations: _Optional[_Iterable[_Union[TraceLocation, _Mapping]]] = ..., filter: _Optional[str] = ..., max_results: _Optional[int] = ..., order_by: _Optional[_Iterable[str]] = ..., sql_warehouse_id: _Optional[str] = ..., page_token: _Optional[str] = ...) -> None: ...

class CreateTraceUCStorageLocation(_message.Message):
    __slots__ = ("uc_schema", "sql_warehouse_id")
    class Response(_message.Message):
        __slots__ = ("uc_schema",)
        UC_SCHEMA_FIELD_NUMBER: _ClassVar[int]
        uc_schema: UCSchemaLocation
        def __init__(self, uc_schema: _Optional[_Union[UCSchemaLocation, _Mapping]] = ...) -> None: ...
    UC_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    SQL_WAREHOUSE_ID_FIELD_NUMBER: _ClassVar[int]
    uc_schema: UCSchemaLocation
    sql_warehouse_id: str
    def __init__(self, uc_schema: _Optional[_Union[UCSchemaLocation, _Mapping]] = ..., sql_warehouse_id: _Optional[str] = ...) -> None: ...

class LinkExperimentToUCTraceLocation(_message.Message):
    __slots__ = ("experiment_id", "uc_schema")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    UC_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    uc_schema: UCSchemaLocation
    def __init__(self, experiment_id: _Optional[str] = ..., uc_schema: _Optional[_Union[UCSchemaLocation, _Mapping]] = ...) -> None: ...

class UnLinkExperimentToUCTraceLocation(_message.Message):
    __slots__ = ("experiment_id", "uc_schema")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    UC_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    uc_schema: UCSchemaLocation
    def __init__(self, experiment_id: _Optional[str] = ..., uc_schema: _Optional[_Union[UCSchemaLocation, _Mapping]] = ...) -> None: ...

class Assessment(_message.Message):
    __slots__ = ("assessment_id", "assessment_name", "trace_id", "trace_location", "span_id", "source", "create_time", "last_update_time", "feedback", "expectation", "rationale", "metadata", "overrides", "valid")
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
    TRACE_LOCATION_FIELD_NUMBER: _ClassVar[int]
    SPAN_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    CREATE_TIME_FIELD_NUMBER: _ClassVar[int]
    LAST_UPDATE_TIME_FIELD_NUMBER: _ClassVar[int]
    FEEDBACK_FIELD_NUMBER: _ClassVar[int]
    EXPECTATION_FIELD_NUMBER: _ClassVar[int]
    RATIONALE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    OVERRIDES_FIELD_NUMBER: _ClassVar[int]
    VALID_FIELD_NUMBER: _ClassVar[int]
    assessment_id: str
    assessment_name: str
    trace_id: str
    trace_location: TraceLocation
    span_id: str
    source: _assessments_pb2.AssessmentSource
    create_time: _timestamp_pb2.Timestamp
    last_update_time: _timestamp_pb2.Timestamp
    feedback: _assessments_pb2.Feedback
    expectation: _assessments_pb2.Expectation
    rationale: str
    metadata: _containers.ScalarMap[str, str]
    overrides: str
    valid: bool
    def __init__(self, assessment_id: _Optional[str] = ..., assessment_name: _Optional[str] = ..., trace_id: _Optional[str] = ..., trace_location: _Optional[_Union[TraceLocation, _Mapping]] = ..., span_id: _Optional[str] = ..., source: _Optional[_Union[_assessments_pb2.AssessmentSource, _Mapping]] = ..., create_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., last_update_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., feedback: _Optional[_Union[_assessments_pb2.Feedback, _Mapping]] = ..., expectation: _Optional[_Union[_assessments_pb2.Expectation, _Mapping]] = ..., rationale: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ..., overrides: _Optional[str] = ..., valid: bool = ...) -> None: ...

class CreateAssessment(_message.Message):
    __slots__ = ("location_id", "assessment", "sql_warehouse_id")
    class Response(_message.Message):
        __slots__ = ("assessment",)
        ASSESSMENT_FIELD_NUMBER: _ClassVar[int]
        assessment: Assessment
        def __init__(self, assessment: _Optional[_Union[Assessment, _Mapping]] = ...) -> None: ...
    LOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    ASSESSMENT_FIELD_NUMBER: _ClassVar[int]
    SQL_WAREHOUSE_ID_FIELD_NUMBER: _ClassVar[int]
    location_id: str
    assessment: Assessment
    sql_warehouse_id: str
    def __init__(self, location_id: _Optional[str] = ..., assessment: _Optional[_Union[Assessment, _Mapping]] = ..., sql_warehouse_id: _Optional[str] = ...) -> None: ...

class GetAssessment(_message.Message):
    __slots__ = ("location_id", "trace_id", "assessment_id", "sql_warehouse_id")
    class Response(_message.Message):
        __slots__ = ("assessment",)
        ASSESSMENT_FIELD_NUMBER: _ClassVar[int]
        assessment: Assessment
        def __init__(self, assessment: _Optional[_Union[Assessment, _Mapping]] = ...) -> None: ...
    LOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    ASSESSMENT_ID_FIELD_NUMBER: _ClassVar[int]
    SQL_WAREHOUSE_ID_FIELD_NUMBER: _ClassVar[int]
    location_id: str
    trace_id: str
    assessment_id: str
    sql_warehouse_id: str
    def __init__(self, location_id: _Optional[str] = ..., trace_id: _Optional[str] = ..., assessment_id: _Optional[str] = ..., sql_warehouse_id: _Optional[str] = ...) -> None: ...

class UpdateAssessment(_message.Message):
    __slots__ = ("location_id", "assessment", "update_mask", "sql_warehouse_id")
    class Response(_message.Message):
        __slots__ = ("assessment",)
        ASSESSMENT_FIELD_NUMBER: _ClassVar[int]
        assessment: Assessment
        def __init__(self, assessment: _Optional[_Union[Assessment, _Mapping]] = ...) -> None: ...
    LOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    ASSESSMENT_FIELD_NUMBER: _ClassVar[int]
    UPDATE_MASK_FIELD_NUMBER: _ClassVar[int]
    SQL_WAREHOUSE_ID_FIELD_NUMBER: _ClassVar[int]
    location_id: str
    assessment: Assessment
    update_mask: _field_mask_pb2.FieldMask
    sql_warehouse_id: str
    def __init__(self, location_id: _Optional[str] = ..., assessment: _Optional[_Union[Assessment, _Mapping]] = ..., update_mask: _Optional[_Union[_field_mask_pb2.FieldMask, _Mapping]] = ..., sql_warehouse_id: _Optional[str] = ...) -> None: ...

class DeleteAssessment(_message.Message):
    __slots__ = ("location_id", "trace_id", "assessment_id", "sql_warehouse_id")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    LOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    ASSESSMENT_ID_FIELD_NUMBER: _ClassVar[int]
    SQL_WAREHOUSE_ID_FIELD_NUMBER: _ClassVar[int]
    location_id: str
    trace_id: str
    assessment_id: str
    sql_warehouse_id: str
    def __init__(self, location_id: _Optional[str] = ..., trace_id: _Optional[str] = ..., assessment_id: _Optional[str] = ..., sql_warehouse_id: _Optional[str] = ...) -> None: ...

class BatchLinkTraceToRun(_message.Message):
    __slots__ = ("location_id", "trace_ids", "run_id")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    LOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    TRACE_IDS_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    location_id: str
    trace_ids: _containers.RepeatedScalarFieldContainer[str]
    run_id: str
    def __init__(self, location_id: _Optional[str] = ..., trace_ids: _Optional[_Iterable[str]] = ..., run_id: _Optional[str] = ...) -> None: ...

class BatchUnlinkTraceFromRun(_message.Message):
    __slots__ = ("location_id", "trace_ids", "run_id")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    LOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    TRACE_IDS_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    location_id: str
    trace_ids: _containers.RepeatedScalarFieldContainer[str]
    run_id: str
    def __init__(self, location_id: _Optional[str] = ..., trace_ids: _Optional[_Iterable[str]] = ..., run_id: _Optional[str] = ...) -> None: ...

class DatabricksTrackingService(_service.service): ...

class DatabricksTrackingService_Stub(DatabricksTrackingService): ...

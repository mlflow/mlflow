from scalapb import scalapb_pb2 as _scalapb_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class JobStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    JOB_STATUS_UNSPECIFIED: _ClassVar[JobStatus]
    JOB_STATUS_PENDING: _ClassVar[JobStatus]
    JOB_STATUS_IN_PROGRESS: _ClassVar[JobStatus]
    JOB_STATUS_COMPLETED: _ClassVar[JobStatus]
    JOB_STATUS_FAILED: _ClassVar[JobStatus]
    JOB_STATUS_CANCELED: _ClassVar[JobStatus]
    JOB_STATUS_NEEDS_RECOVERY: _ClassVar[JobStatus]
JOB_STATUS_UNSPECIFIED: JobStatus
JOB_STATUS_PENDING: JobStatus
JOB_STATUS_IN_PROGRESS: JobStatus
JOB_STATUS_COMPLETED: JobStatus
JOB_STATUS_FAILED: JobStatus
JOB_STATUS_CANCELED: JobStatus
JOB_STATUS_NEEDS_RECOVERY: JobStatus

class JobProgress(_message.Message):
    __slots__ = ("phase", "completed", "total", "unit")
    PHASE_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    phase: str
    completed: int
    total: int
    unit: str
    def __init__(self, phase: _Optional[str] = ..., completed: _Optional[int] = ..., total: _Optional[int] = ..., unit: _Optional[str] = ...) -> None: ...

class JobState(_message.Message):
    __slots__ = ("status", "error_message", "metadata", "status_message", "progress_payload", "progress_updated_at")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    STATUS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    status: JobStatus
    error_message: str
    metadata: _containers.ScalarMap[str, str]
    status_message: str
    progress_payload: JobProgress
    progress_updated_at: int
    def __init__(self, status: _Optional[_Union[JobStatus, str]] = ..., error_message: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ..., status_message: _Optional[str] = ..., progress_payload: _Optional[_Union[JobProgress, _Mapping]] = ..., progress_updated_at: _Optional[int] = ...) -> None: ...

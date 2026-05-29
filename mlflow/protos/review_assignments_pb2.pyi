import databricks_pb2 as _databricks_pb2
from scalapb import scalapb_pb2 as _scalapb_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ReviewTargetType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    REVIEW_TARGET_TYPE_UNSPECIFIED: _ClassVar[ReviewTargetType]
    TRACE: _ClassVar[ReviewTargetType]

class ReviewAssignmentState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    REVIEW_ASSIGNMENT_STATE_UNSPECIFIED: _ClassVar[ReviewAssignmentState]
    PENDING: _ClassVar[ReviewAssignmentState]
    IN_PROGRESS: _ClassVar[ReviewAssignmentState]
    COMPLETE: _ClassVar[ReviewAssignmentState]
REVIEW_TARGET_TYPE_UNSPECIFIED: ReviewTargetType
TRACE: ReviewTargetType
REVIEW_ASSIGNMENT_STATE_UNSPECIFIED: ReviewAssignmentState
PENDING: ReviewAssignmentState
IN_PROGRESS: ReviewAssignmentState
COMPLETE: ReviewAssignmentState

class ReviewAssignment(_message.Message):
    __slots__ = ("assignment_id", "experiment_id", "target_type", "target_id", "reviewer", "assigner", "state", "creation_time_ms", "last_update_time_ms", "completed_time_ms")
    ASSIGNMENT_ID_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_TYPE_FIELD_NUMBER: _ClassVar[int]
    TARGET_ID_FIELD_NUMBER: _ClassVar[int]
    REVIEWER_FIELD_NUMBER: _ClassVar[int]
    ASSIGNER_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    CREATION_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    LAST_UPDATE_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    assignment_id: str
    experiment_id: str
    target_type: ReviewTargetType
    target_id: str
    reviewer: str
    assigner: str
    state: ReviewAssignmentState
    creation_time_ms: int
    last_update_time_ms: int
    completed_time_ms: int
    def __init__(self, assignment_id: _Optional[str] = ..., experiment_id: _Optional[str] = ..., target_type: _Optional[_Union[ReviewTargetType, str]] = ..., target_id: _Optional[str] = ..., reviewer: _Optional[str] = ..., assigner: _Optional[str] = ..., state: _Optional[_Union[ReviewAssignmentState, str]] = ..., creation_time_ms: _Optional[int] = ..., last_update_time_ms: _Optional[int] = ..., completed_time_ms: _Optional[int] = ...) -> None: ...

class BulkCreateFailure(_message.Message):
    __slots__ = ("target_id", "reviewer", "error_message")
    TARGET_ID_FIELD_NUMBER: _ClassVar[int]
    REVIEWER_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    target_id: str
    reviewer: str
    error_message: str
    def __init__(self, target_id: _Optional[str] = ..., reviewer: _Optional[str] = ..., error_message: _Optional[str] = ...) -> None: ...

class CreateReviewAssignment(_message.Message):
    __slots__ = ("experiment_id", "target_type", "target_id", "reviewer", "assigner")
    class Response(_message.Message):
        __slots__ = ("review_assignment",)
        REVIEW_ASSIGNMENT_FIELD_NUMBER: _ClassVar[int]
        review_assignment: ReviewAssignment
        def __init__(self, review_assignment: _Optional[_Union[ReviewAssignment, _Mapping]] = ...) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_TYPE_FIELD_NUMBER: _ClassVar[int]
    TARGET_ID_FIELD_NUMBER: _ClassVar[int]
    REVIEWER_FIELD_NUMBER: _ClassVar[int]
    ASSIGNER_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    target_type: ReviewTargetType
    target_id: str
    reviewer: str
    assigner: str
    def __init__(self, experiment_id: _Optional[str] = ..., target_type: _Optional[_Union[ReviewTargetType, str]] = ..., target_id: _Optional[str] = ..., reviewer: _Optional[str] = ..., assigner: _Optional[str] = ...) -> None: ...

class BulkCreateReviewAssignments(_message.Message):
    __slots__ = ("experiment_id", "target_type", "target_ids", "reviewers", "assigner")
    class Response(_message.Message):
        __slots__ = ("created", "existing", "failed")
        CREATED_FIELD_NUMBER: _ClassVar[int]
        EXISTING_FIELD_NUMBER: _ClassVar[int]
        FAILED_FIELD_NUMBER: _ClassVar[int]
        created: _containers.RepeatedCompositeFieldContainer[ReviewAssignment]
        existing: _containers.RepeatedScalarFieldContainer[str]
        failed: _containers.RepeatedCompositeFieldContainer[BulkCreateFailure]
        def __init__(self, created: _Optional[_Iterable[_Union[ReviewAssignment, _Mapping]]] = ..., existing: _Optional[_Iterable[str]] = ..., failed: _Optional[_Iterable[_Union[BulkCreateFailure, _Mapping]]] = ...) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_TYPE_FIELD_NUMBER: _ClassVar[int]
    TARGET_IDS_FIELD_NUMBER: _ClassVar[int]
    REVIEWERS_FIELD_NUMBER: _ClassVar[int]
    ASSIGNER_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    target_type: ReviewTargetType
    target_ids: _containers.RepeatedScalarFieldContainer[str]
    reviewers: _containers.RepeatedScalarFieldContainer[str]
    assigner: str
    def __init__(self, experiment_id: _Optional[str] = ..., target_type: _Optional[_Union[ReviewTargetType, str]] = ..., target_ids: _Optional[_Iterable[str]] = ..., reviewers: _Optional[_Iterable[str]] = ..., assigner: _Optional[str] = ...) -> None: ...

class GetReviewAssignment(_message.Message):
    __slots__ = ("assignment_id",)
    class Response(_message.Message):
        __slots__ = ("review_assignment",)
        REVIEW_ASSIGNMENT_FIELD_NUMBER: _ClassVar[int]
        review_assignment: ReviewAssignment
        def __init__(self, review_assignment: _Optional[_Union[ReviewAssignment, _Mapping]] = ...) -> None: ...
    ASSIGNMENT_ID_FIELD_NUMBER: _ClassVar[int]
    assignment_id: str
    def __init__(self, assignment_id: _Optional[str] = ...) -> None: ...

class ListReviewAssignments(_message.Message):
    __slots__ = ("experiment_id", "reviewer", "state", "target_type", "max_results", "page_token")
    class Response(_message.Message):
        __slots__ = ("review_assignments", "next_page_token")
        REVIEW_ASSIGNMENTS_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        review_assignments: _containers.RepeatedCompositeFieldContainer[ReviewAssignment]
        next_page_token: str
        def __init__(self, review_assignments: _Optional[_Iterable[_Union[ReviewAssignment, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    REVIEWER_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    TARGET_TYPE_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    reviewer: str
    state: ReviewAssignmentState
    target_type: ReviewTargetType
    max_results: int
    page_token: str
    def __init__(self, experiment_id: _Optional[str] = ..., reviewer: _Optional[str] = ..., state: _Optional[_Union[ReviewAssignmentState, str]] = ..., target_type: _Optional[_Union[ReviewTargetType, str]] = ..., max_results: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...

class ListReviewAssignmentsForTarget(_message.Message):
    __slots__ = ("target_id",)
    class Response(_message.Message):
        __slots__ = ("review_assignments",)
        REVIEW_ASSIGNMENTS_FIELD_NUMBER: _ClassVar[int]
        review_assignments: _containers.RepeatedCompositeFieldContainer[ReviewAssignment]
        def __init__(self, review_assignments: _Optional[_Iterable[_Union[ReviewAssignment, _Mapping]]] = ...) -> None: ...
    TARGET_ID_FIELD_NUMBER: _ClassVar[int]
    target_id: str
    def __init__(self, target_id: _Optional[str] = ...) -> None: ...

class UpdateReviewAssignment(_message.Message):
    __slots__ = ("assignment_id", "state")
    class Response(_message.Message):
        __slots__ = ("review_assignment",)
        REVIEW_ASSIGNMENT_FIELD_NUMBER: _ClassVar[int]
        review_assignment: ReviewAssignment
        def __init__(self, review_assignment: _Optional[_Union[ReviewAssignment, _Mapping]] = ...) -> None: ...
    ASSIGNMENT_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    assignment_id: str
    state: ReviewAssignmentState
    def __init__(self, assignment_id: _Optional[str] = ..., state: _Optional[_Union[ReviewAssignmentState, str]] = ...) -> None: ...

class DeleteReviewAssignment(_message.Message):
    __slots__ = ("assignment_id",)
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    ASSIGNMENT_ID_FIELD_NUMBER: _ClassVar[int]
    assignment_id: str
    def __init__(self, assignment_id: _Optional[str] = ...) -> None: ...

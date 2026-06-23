import databricks_pb2 as _databricks_pb2
from scalapb import scalapb_pb2 as _scalapb_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ReviewItemType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    REVIEW_ITEM_TYPE_UNSPECIFIED: _ClassVar[ReviewItemType]
    TRACE: _ClassVar[ReviewItemType]

class ReviewQueueType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    REVIEW_QUEUE_TYPE_UNSPECIFIED: _ClassVar[ReviewQueueType]
    USER: _ClassVar[ReviewQueueType]
    CUSTOM: _ClassVar[ReviewQueueType]

class ReviewStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    REVIEW_STATUS_UNSPECIFIED: _ClassVar[ReviewStatus]
    PENDING: _ClassVar[ReviewStatus]
    COMPLETE: _ClassVar[ReviewStatus]
    DECLINED: _ClassVar[ReviewStatus]
REVIEW_ITEM_TYPE_UNSPECIFIED: ReviewItemType
TRACE: ReviewItemType
REVIEW_QUEUE_TYPE_UNSPECIFIED: ReviewQueueType
USER: ReviewQueueType
CUSTOM: ReviewQueueType
REVIEW_STATUS_UNSPECIFIED: ReviewStatus
PENDING: ReviewStatus
COMPLETE: ReviewStatus
DECLINED: ReviewStatus

class ReviewQueue(_message.Message):
    __slots__ = ("queue_id", "experiment_id", "name", "queue_type", "created_by", "creation_time_ms", "last_update_time_ms", "users", "schema_ids")
    QUEUE_ID_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    QUEUE_TYPE_FIELD_NUMBER: _ClassVar[int]
    CREATED_BY_FIELD_NUMBER: _ClassVar[int]
    CREATION_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    LAST_UPDATE_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    USERS_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_IDS_FIELD_NUMBER: _ClassVar[int]
    queue_id: str
    experiment_id: str
    name: str
    queue_type: ReviewQueueType
    created_by: str
    creation_time_ms: int
    last_update_time_ms: int
    users: _containers.RepeatedScalarFieldContainer[str]
    schema_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, queue_id: _Optional[str] = ..., experiment_id: _Optional[str] = ..., name: _Optional[str] = ..., queue_type: _Optional[_Union[ReviewQueueType, str]] = ..., created_by: _Optional[str] = ..., creation_time_ms: _Optional[int] = ..., last_update_time_ms: _Optional[int] = ..., users: _Optional[_Iterable[str]] = ..., schema_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class ReviewQueueItem(_message.Message):
    __slots__ = ("queue_id", "item_type", "item_id", "status", "completed_by", "completed_time_ms", "creation_time_ms", "last_update_time_ms")
    QUEUE_ID_FIELD_NUMBER: _ClassVar[int]
    ITEM_TYPE_FIELD_NUMBER: _ClassVar[int]
    ITEM_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_BY_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    CREATION_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    LAST_UPDATE_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    queue_id: str
    item_type: ReviewItemType
    item_id: str
    status: ReviewStatus
    completed_by: str
    completed_time_ms: int
    creation_time_ms: int
    last_update_time_ms: int
    def __init__(self, queue_id: _Optional[str] = ..., item_type: _Optional[_Union[ReviewItemType, str]] = ..., item_id: _Optional[str] = ..., status: _Optional[_Union[ReviewStatus, str]] = ..., completed_by: _Optional[str] = ..., completed_time_ms: _Optional[int] = ..., creation_time_ms: _Optional[int] = ..., last_update_time_ms: _Optional[int] = ...) -> None: ...

class CreateReviewQueue(_message.Message):
    __slots__ = ("experiment_id", "name", "queue_type", "created_by", "users", "schema_ids")
    class Response(_message.Message):
        __slots__ = ("review_queue",)
        REVIEW_QUEUE_FIELD_NUMBER: _ClassVar[int]
        review_queue: ReviewQueue
        def __init__(self, review_queue: _Optional[_Union[ReviewQueue, _Mapping]] = ...) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    QUEUE_TYPE_FIELD_NUMBER: _ClassVar[int]
    CREATED_BY_FIELD_NUMBER: _ClassVar[int]
    USERS_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_IDS_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    name: str
    queue_type: ReviewQueueType
    created_by: str
    users: _containers.RepeatedScalarFieldContainer[str]
    schema_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, experiment_id: _Optional[str] = ..., name: _Optional[str] = ..., queue_type: _Optional[_Union[ReviewQueueType, str]] = ..., created_by: _Optional[str] = ..., users: _Optional[_Iterable[str]] = ..., schema_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class GetOrCreateUserQueue(_message.Message):
    __slots__ = ("experiment_id", "user", "created_by")
    class Response(_message.Message):
        __slots__ = ("review_queue",)
        REVIEW_QUEUE_FIELD_NUMBER: _ClassVar[int]
        review_queue: ReviewQueue
        def __init__(self, review_queue: _Optional[_Union[ReviewQueue, _Mapping]] = ...) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    USER_FIELD_NUMBER: _ClassVar[int]
    CREATED_BY_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    user: str
    created_by: str
    def __init__(self, experiment_id: _Optional[str] = ..., user: _Optional[str] = ..., created_by: _Optional[str] = ...) -> None: ...

class GetReviewQueue(_message.Message):
    __slots__ = ("queue_id",)
    class Response(_message.Message):
        __slots__ = ("review_queue",)
        REVIEW_QUEUE_FIELD_NUMBER: _ClassVar[int]
        review_queue: ReviewQueue
        def __init__(self, review_queue: _Optional[_Union[ReviewQueue, _Mapping]] = ...) -> None: ...
    QUEUE_ID_FIELD_NUMBER: _ClassVar[int]
    queue_id: str
    def __init__(self, queue_id: _Optional[str] = ...) -> None: ...

class GetReviewQueueByName(_message.Message):
    __slots__ = ("experiment_id", "name")
    class Response(_message.Message):
        __slots__ = ("review_queue",)
        REVIEW_QUEUE_FIELD_NUMBER: _ClassVar[int]
        review_queue: ReviewQueue
        def __init__(self, review_queue: _Optional[_Union[ReviewQueue, _Mapping]] = ...) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    name: str
    def __init__(self, experiment_id: _Optional[str] = ..., name: _Optional[str] = ...) -> None: ...

class ListReviewQueues(_message.Message):
    __slots__ = ("experiment_id", "user", "max_results", "page_token", "item_id")
    class Response(_message.Message):
        __slots__ = ("review_queues", "next_page_token")
        REVIEW_QUEUES_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        review_queues: _containers.RepeatedCompositeFieldContainer[ReviewQueue]
        next_page_token: str
        def __init__(self, review_queues: _Optional[_Iterable[_Union[ReviewQueue, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    USER_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    ITEM_ID_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    user: str
    max_results: int
    page_token: str
    item_id: str
    def __init__(self, experiment_id: _Optional[str] = ..., user: _Optional[str] = ..., max_results: _Optional[int] = ..., page_token: _Optional[str] = ..., item_id: _Optional[str] = ...) -> None: ...

class UpdateReviewQueue(_message.Message):
    __slots__ = ("queue_id", "update_users", "users", "update_schema_ids", "schema_ids", "name", "new_owner")
    class Response(_message.Message):
        __slots__ = ("review_queue",)
        REVIEW_QUEUE_FIELD_NUMBER: _ClassVar[int]
        review_queue: ReviewQueue
        def __init__(self, review_queue: _Optional[_Union[ReviewQueue, _Mapping]] = ...) -> None: ...
    QUEUE_ID_FIELD_NUMBER: _ClassVar[int]
    UPDATE_USERS_FIELD_NUMBER: _ClassVar[int]
    USERS_FIELD_NUMBER: _ClassVar[int]
    UPDATE_SCHEMA_IDS_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_IDS_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    NEW_OWNER_FIELD_NUMBER: _ClassVar[int]
    queue_id: str
    update_users: bool
    users: _containers.RepeatedScalarFieldContainer[str]
    update_schema_ids: bool
    schema_ids: _containers.RepeatedScalarFieldContainer[str]
    name: str
    new_owner: str
    def __init__(self, queue_id: _Optional[str] = ..., update_users: bool = ..., users: _Optional[_Iterable[str]] = ..., update_schema_ids: bool = ..., schema_ids: _Optional[_Iterable[str]] = ..., name: _Optional[str] = ..., new_owner: _Optional[str] = ...) -> None: ...

class DeleteReviewQueue(_message.Message):
    __slots__ = ("queue_id",)
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    QUEUE_ID_FIELD_NUMBER: _ClassVar[int]
    queue_id: str
    def __init__(self, queue_id: _Optional[str] = ...) -> None: ...

class AddItemsToReviewQueue(_message.Message):
    __slots__ = ("queue_id", "item_type", "item_ids")
    class Response(_message.Message):
        __slots__ = ("items",)
        ITEMS_FIELD_NUMBER: _ClassVar[int]
        items: _containers.RepeatedCompositeFieldContainer[ReviewQueueItem]
        def __init__(self, items: _Optional[_Iterable[_Union[ReviewQueueItem, _Mapping]]] = ...) -> None: ...
    QUEUE_ID_FIELD_NUMBER: _ClassVar[int]
    ITEM_TYPE_FIELD_NUMBER: _ClassVar[int]
    ITEM_IDS_FIELD_NUMBER: _ClassVar[int]
    queue_id: str
    item_type: ReviewItemType
    item_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, queue_id: _Optional[str] = ..., item_type: _Optional[_Union[ReviewItemType, str]] = ..., item_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class RemoveItemsFromReviewQueue(_message.Message):
    __slots__ = ("queue_id", "item_ids")
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    QUEUE_ID_FIELD_NUMBER: _ClassVar[int]
    ITEM_IDS_FIELD_NUMBER: _ClassVar[int]
    queue_id: str
    item_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, queue_id: _Optional[str] = ..., item_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class ListReviewQueueItems(_message.Message):
    __slots__ = ("queue_id", "status", "max_results", "page_token")
    class Response(_message.Message):
        __slots__ = ("items", "next_page_token")
        ITEMS_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        items: _containers.RepeatedCompositeFieldContainer[ReviewQueueItem]
        next_page_token: str
        def __init__(self, items: _Optional[_Iterable[_Union[ReviewQueueItem, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    QUEUE_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    queue_id: str
    status: ReviewStatus
    max_results: int
    page_token: str
    def __init__(self, queue_id: _Optional[str] = ..., status: _Optional[_Union[ReviewStatus, str]] = ..., max_results: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...

class SetReviewQueueItemStatus(_message.Message):
    __slots__ = ("queue_id", "item_id", "status", "completed_by")
    class Response(_message.Message):
        __slots__ = ("item",)
        ITEM_FIELD_NUMBER: _ClassVar[int]
        item: ReviewQueueItem
        def __init__(self, item: _Optional[_Union[ReviewQueueItem, _Mapping]] = ...) -> None: ...
    QUEUE_ID_FIELD_NUMBER: _ClassVar[int]
    ITEM_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_BY_FIELD_NUMBER: _ClassVar[int]
    queue_id: str
    item_id: str
    status: ReviewStatus
    completed_by: str
    def __init__(self, queue_id: _Optional[str] = ..., item_id: _Optional[str] = ..., status: _Optional[_Union[ReviewStatus, str]] = ..., completed_by: _Optional[str] = ...) -> None: ...

import databricks_pb2 as _databricks_pb2
from scalapb import scalapb_pb2 as _scalapb_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import service as _service
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class WebhookStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTIVE: _ClassVar[WebhookStatus]
    DISABLED: _ClassVar[WebhookStatus]

class WebhookEntity(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ENTITY_UNSPECIFIED: _ClassVar[WebhookEntity]
    REGISTERED_MODEL: _ClassVar[WebhookEntity]
    MODEL_VERSION: _ClassVar[WebhookEntity]
    MODEL_VERSION_TAG: _ClassVar[WebhookEntity]
    MODEL_VERSION_ALIAS: _ClassVar[WebhookEntity]
    PROMPT: _ClassVar[WebhookEntity]
    PROMPT_VERSION: _ClassVar[WebhookEntity]
    PROMPT_TAG: _ClassVar[WebhookEntity]
    PROMPT_VERSION_TAG: _ClassVar[WebhookEntity]
    PROMPT_ALIAS: _ClassVar[WebhookEntity]

class WebhookAction(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTION_UNSPECIFIED: _ClassVar[WebhookAction]
    CREATED: _ClassVar[WebhookAction]
    UPDATED: _ClassVar[WebhookAction]
    DELETED: _ClassVar[WebhookAction]
    SET: _ClassVar[WebhookAction]
ACTIVE: WebhookStatus
DISABLED: WebhookStatus
ENTITY_UNSPECIFIED: WebhookEntity
REGISTERED_MODEL: WebhookEntity
MODEL_VERSION: WebhookEntity
MODEL_VERSION_TAG: WebhookEntity
MODEL_VERSION_ALIAS: WebhookEntity
PROMPT: WebhookEntity
PROMPT_VERSION: WebhookEntity
PROMPT_TAG: WebhookEntity
PROMPT_VERSION_TAG: WebhookEntity
PROMPT_ALIAS: WebhookEntity
ACTION_UNSPECIFIED: WebhookAction
CREATED: WebhookAction
UPDATED: WebhookAction
DELETED: WebhookAction
SET: WebhookAction

class WebhookEvent(_message.Message):
    __slots__ = ("entity", "action")
    ENTITY_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    entity: WebhookEntity
    action: WebhookAction
    def __init__(self, entity: _Optional[_Union[WebhookEntity, str]] = ..., action: _Optional[_Union[WebhookAction, str]] = ...) -> None: ...

class Webhook(_message.Message):
    __slots__ = ("webhook_id", "name", "description", "url", "events", "status", "creation_timestamp", "last_updated_timestamp")
    WEBHOOK_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CREATION_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    LAST_UPDATED_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    webhook_id: str
    name: str
    description: str
    url: str
    events: _containers.RepeatedCompositeFieldContainer[WebhookEvent]
    status: WebhookStatus
    creation_timestamp: int
    last_updated_timestamp: int
    def __init__(self, webhook_id: _Optional[str] = ..., name: _Optional[str] = ..., description: _Optional[str] = ..., url: _Optional[str] = ..., events: _Optional[_Iterable[_Union[WebhookEvent, _Mapping]]] = ..., status: _Optional[_Union[WebhookStatus, str]] = ..., creation_timestamp: _Optional[int] = ..., last_updated_timestamp: _Optional[int] = ...) -> None: ...

class WebhookTestResult(_message.Message):
    __slots__ = ("success", "response_status", "response_body", "error_message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_STATUS_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_BODY_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    response_status: int
    response_body: str
    error_message: str
    def __init__(self, success: bool = ..., response_status: _Optional[int] = ..., response_body: _Optional[str] = ..., error_message: _Optional[str] = ...) -> None: ...

class CreateWebhook(_message.Message):
    __slots__ = ("name", "description", "url", "events", "secret", "status")
    class Response(_message.Message):
        __slots__ = ("webhook",)
        WEBHOOK_FIELD_NUMBER: _ClassVar[int]
        webhook: Webhook
        def __init__(self, webhook: _Optional[_Union[Webhook, _Mapping]] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    SECRET_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    name: str
    description: str
    url: str
    events: _containers.RepeatedCompositeFieldContainer[WebhookEvent]
    secret: str
    status: WebhookStatus
    def __init__(self, name: _Optional[str] = ..., description: _Optional[str] = ..., url: _Optional[str] = ..., events: _Optional[_Iterable[_Union[WebhookEvent, _Mapping]]] = ..., secret: _Optional[str] = ..., status: _Optional[_Union[WebhookStatus, str]] = ...) -> None: ...

class ListWebhooks(_message.Message):
    __slots__ = ("max_results", "page_token")
    class Response(_message.Message):
        __slots__ = ("webhooks", "next_page_token")
        WEBHOOKS_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        webhooks: _containers.RepeatedCompositeFieldContainer[Webhook]
        next_page_token: str
        def __init__(self, webhooks: _Optional[_Iterable[_Union[Webhook, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    max_results: int
    page_token: str
    def __init__(self, max_results: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...

class GetWebhook(_message.Message):
    __slots__ = ("webhook_id",)
    class Response(_message.Message):
        __slots__ = ("webhook",)
        WEBHOOK_FIELD_NUMBER: _ClassVar[int]
        webhook: Webhook
        def __init__(self, webhook: _Optional[_Union[Webhook, _Mapping]] = ...) -> None: ...
    WEBHOOK_ID_FIELD_NUMBER: _ClassVar[int]
    webhook_id: str
    def __init__(self, webhook_id: _Optional[str] = ...) -> None: ...

class UpdateWebhook(_message.Message):
    __slots__ = ("webhook_id", "name", "description", "url", "events", "secret", "status")
    class Response(_message.Message):
        __slots__ = ("webhook",)
        WEBHOOK_FIELD_NUMBER: _ClassVar[int]
        webhook: Webhook
        def __init__(self, webhook: _Optional[_Union[Webhook, _Mapping]] = ...) -> None: ...
    WEBHOOK_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    SECRET_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    webhook_id: str
    name: str
    description: str
    url: str
    events: _containers.RepeatedCompositeFieldContainer[WebhookEvent]
    secret: str
    status: WebhookStatus
    def __init__(self, webhook_id: _Optional[str] = ..., name: _Optional[str] = ..., description: _Optional[str] = ..., url: _Optional[str] = ..., events: _Optional[_Iterable[_Union[WebhookEvent, _Mapping]]] = ..., secret: _Optional[str] = ..., status: _Optional[_Union[WebhookStatus, str]] = ...) -> None: ...

class DeleteWebhook(_message.Message):
    __slots__ = ("webhook_id",)
    class Response(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    WEBHOOK_ID_FIELD_NUMBER: _ClassVar[int]
    webhook_id: str
    def __init__(self, webhook_id: _Optional[str] = ...) -> None: ...

class TestWebhook(_message.Message):
    __slots__ = ("webhook_id", "event")
    class Response(_message.Message):
        __slots__ = ("result",)
        RESULT_FIELD_NUMBER: _ClassVar[int]
        result: WebhookTestResult
        def __init__(self, result: _Optional[_Union[WebhookTestResult, _Mapping]] = ...) -> None: ...
    WEBHOOK_ID_FIELD_NUMBER: _ClassVar[int]
    EVENT_FIELD_NUMBER: _ClassVar[int]
    webhook_id: str
    event: WebhookEvent
    def __init__(self, webhook_id: _Optional[str] = ..., event: _Optional[_Union[WebhookEvent, _Mapping]] = ...) -> None: ...

class WebhookService(_service.service): ...

class WebhookService_Stub(WebhookService): ...

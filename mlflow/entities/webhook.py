from enum import Enum
from typing import Optional, Union

from mlflow.exceptions import MlflowException
from mlflow.protos.webhooks_pb2 import Webhook as ProtoWebhook
from mlflow.protos.webhooks_pb2 import WebhookEvent as ProtoWebhookEvent
from mlflow.protos.webhooks_pb2 import WebhookStatus as ProtoWebhookStatus
from mlflow.protos.webhooks_pb2 import WebhookTestResult as ProtoWebhookTestResult


class WebhookStatus(str, Enum):
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"

    @classmethod
    def from_proto(cls, proto: int) -> "WebhookStatus":
        return WebhookStatus(ProtoWebhookStatus.Name(proto))

    def to_proto(self) -> int:
        return ProtoWebhookStatus.Value(self.name)


class WebhookEvent(str, Enum):
    # Registered Model Events
    REGISTERED_MODEL_CREATED = "REGISTERED_MODEL_CREATED"

    # Model Version Events
    MODEL_VERSION_CREATED = "MODEL_VERSION_CREATED"
    MODEL_VERSION_TAG_SET = "MODEL_VERSION_TAG_SET"
    MODEL_VERSION_TAG_DELETED = "MODEL_VERSION_TAG_DELETED"

    # Model Version Alias Events
    MODEL_VERSION_ALIAS_CREATED = "MODEL_VERSION_ALIAS_CREATED"
    MODEL_VERSION_ALIAS_DELETED = "MODEL_VERSION_ALIAS_DELETED"

    @classmethod
    def from_proto(cls, proto: int) -> "WebhookEvent":
        return cls(ProtoWebhookEvent.Name(proto))

    def to_proto(self) -> int:
        return ProtoWebhookEvent.Value(self.value)


class Webhook:
    """
    MLflow entity for Webhook.
    """

    def __init__(
        self,
        webhook_id: str,
        name: str,
        url: str,
        events: list[Union[str, WebhookEvent]],
        creation_timestamp: int,
        last_updated_timestamp: int,
        description: Optional[str] = None,
        status: Union[str, WebhookStatus] = WebhookStatus.ACTIVE,
        secret: Optional[str] = None,
    ):
        """
        Initialize a Webhook entity.

        Args:
            webhook_id: Unique webhook identifier
            name: Human-readable webhook name
            url: Webhook endpoint URL
            events: List of event types that trigger this webhook (strings or WebhookEvent enums)
            creation_timestamp: Creation timestamp in milliseconds since Unix epoch
            last_updated_timestamp: Last update timestamp in milliseconds since Unix epoch
            description: Optional webhook description
            status: Webhook status (ACTIVE or DISABLED)
            secret: Optional secret key for HMAC signature verification
        """
        super().__init__()
        self._webhook_id = webhook_id
        self._name = name
        self._url = url
        if not events:
            raise MlflowException.invalid_parameter_value("Webhook events cannot be empty")
        self._events = [(WebhookEvent(e) if isinstance(e, str) else e) for e in events]
        self._description = description
        self._status = WebhookStatus(status) if isinstance(status, str) else status
        self._secret = secret
        self._creation_timestamp = creation_timestamp
        self._last_updated_timestamp = last_updated_timestamp

    @property
    def webhook_id(self) -> str:
        return self._webhook_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def url(self) -> str:
        return self._url

    @property
    def events(self) -> list[str]:
        return self._events

    @property
    def description(self) -> Optional[str]:
        return self._description

    def description(self, new_description: Optional[str]) -> None:
        self._description = new_description

    @property
    def status(self) -> str:
        return self._status

    @property
    def secret(self) -> Optional[str]:
        return self._secret

    @property
    def creation_timestamp(self) -> int:
        return self._creation_timestamp

    @property
    def last_updated_timestamp(self) -> int:
        return self._last_updated_timestamp

    @classmethod
    def from_proto(cls, proto: ProtoWebhook) -> "Webhook":
        return cls(
            webhook_id=proto.webhook_id,
            name=proto.name,
            url=proto.url,
            events=[WebhookEvent.from_proto(e) for e in proto.events],
            description=proto.description or None,
            status=WebhookStatus.from_proto(proto.status),
            creation_timestamp=proto.creation_timestamp,
            last_updated_timestamp=proto.last_updated_timestamp,
        )

    def to_proto(self):
        return ProtoWebhook(
            webhook_id=self.webhook_id,
            name=self.name,
            url=self.url,
            events=[event.to_proto() for event in self.events],
            description=self.description,
            status=self.status.to_proto(),
            creation_timestamp=self.creation_timestamp,
            last_updated_timestamp=self.last_updated_timestamp,
        )

    def __repr__(self) -> str:
        return (
            f"Webhook("
            f"webhook_id='{self.webhook_id}', "
            f"name='{self.name}', "
            f"url='{self.url}', "
            f"status='{self.status}', "
            f"events={self.events}, "
            f"creation_timestamp={self.creation_timestamp}, "
            f"last_updated_timestamp={self.last_updated_timestamp}"
            f")"
        )


class WebhookTestResult:
    """
    MLflow entity for WebhookTestResult.
    """

    def __init__(
        self,
        success: bool,
        response_status: Optional[int] = None,
        response_body: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        """
        Initialize a WebhookTestResult entity.

        Args:
            success: Whether the test succeeded
            response_status: HTTP response status code if available
            response_body: Response body if available
            error_message: Error message if test failed
        """
        self._success = success
        self._response_status = response_status
        self._response_body = response_body
        self._error_message = error_message

    @property
    def success(self) -> bool:
        return self._success

    @property
    def response_status(self) -> Optional[int]:
        return self._response_status

    @property
    def response_body(self) -> Optional[str]:
        return self._response_body

    @property
    def error_message(self) -> Optional[str]:
        return self._error_message

    @classmethod
    def from_proto(cls, proto: ProtoWebhookTestResult) -> "WebhookTestResult":
        return cls(
            success=proto.success,
            response_status=proto.response_status if proto.HasField("response_status") else None,
            response_body=proto.response_body if proto.HasField("response_body") else None,
            error_message=proto.error_message if proto.HasField("error_message") else None,
        )

    def to_proto(self) -> ProtoWebhookTestResult:
        return ProtoWebhookTestResult(
            success=self.success,
            response_status=self.response_status,
            response_body=self.response_body,
            error_message=self.error_message,
        )

    def __repr__(self) -> str:
        return (
            f"WebhookTestResult("
            f"success={self.success}, "
            f"response_status={self.response_status}, "
            f"error_message='{self.error_message}'"
            f")"
        )

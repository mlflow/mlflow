from enum import Enum
from typing import Literal, TypeAlias

from typing_extensions import Self

from mlflow.exceptions import MlflowException
from mlflow.protos.webhooks_pb2 import Webhook as ProtoWebhook
from mlflow.protos.webhooks_pb2 import WebhookAction as ProtoWebhookAction
from mlflow.protos.webhooks_pb2 import WebhookEntity as ProtoWebhookEntity
from mlflow.protos.webhooks_pb2 import WebhookEvent as ProtoWebhookEvent
from mlflow.protos.webhooks_pb2 import WebhookStatus as ProtoWebhookStatus
from mlflow.protos.webhooks_pb2 import WebhookTestResult as ProtoWebhookTestResult


class WebhookStatus(str, Enum):
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_proto(cls, proto: int) -> Self:
        proto_name = ProtoWebhookStatus.Name(proto)
        try:
            return cls(proto_name)
        except ValueError:
            raise ValueError(f"Unknown proto status: {proto_name}")

    def to_proto(self) -> int:
        return ProtoWebhookStatus.Value(self.value)

    def is_active(self) -> bool:
        return self == WebhookStatus.ACTIVE


class WebhookEntity(str, Enum):
    REGISTERED_MODEL = "registered_model"
    MODEL_VERSION = "model_version"
    MODEL_VERSION_TAG = "model_version_tag"
    MODEL_VERSION_ALIAS = "model_version_alias"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_proto(cls, proto: int) -> Self:
        proto_name = ProtoWebhookEntity.Name(proto)
        entity_value = proto_name.lower()
        return cls(entity_value)

    def to_proto(self) -> int:
        proto_name = self.value.upper()
        return ProtoWebhookEntity.Value(proto_name)


class WebhookAction(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    SET = "set"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_proto(cls, proto: int) -> Self:
        proto_name = ProtoWebhookAction.Name(proto)
        # Convert UPPER_CASE to lowercase
        action_value = proto_name.lower()
        try:
            return cls(action_value)
        except ValueError:
            raise ValueError(f"Unknown proto action: {proto_name}")

    def to_proto(self) -> int:
        # Convert lowercase to UPPER_CASE
        proto_name = self.value.upper()
        return ProtoWebhookAction.Value(proto_name)


WebhookEventStr: TypeAlias = Literal[
    "registered_model.created",
    "model_version.created",
    "model_version_tag.set",
    "model_version_tag.deleted",
    "model_version_alias.created",
    "model_version_alias.deleted",
]

# Valid actions for each entity type
VALID_ENTITY_ACTIONS: dict[WebhookEntity, set[WebhookAction]] = {
    WebhookEntity.REGISTERED_MODEL: {
        WebhookAction.CREATED,
    },
    WebhookEntity.MODEL_VERSION: {
        WebhookAction.CREATED,
    },
    WebhookEntity.MODEL_VERSION_TAG: {
        WebhookAction.SET,
        WebhookAction.DELETED,
    },
    WebhookEntity.MODEL_VERSION_ALIAS: {
        WebhookAction.CREATED,
        WebhookAction.DELETED,
    },
}


class WebhookEvent:
    """
    Represents a webhook event with a resource and action.
    """

    def __init__(
        self,
        entity: str | WebhookEntity,
        action: str | WebhookAction,
    ):
        """
        Initialize a WebhookEvent.

        Args:
            entity: The entity type (string or WebhookEntity enum)
            action: The action type (string or WebhookAction enum)

        Raises:
            MlflowException: If the entity/action combination is invalid
        """
        self._entity = WebhookEntity(entity) if isinstance(entity, str) else entity
        self._action = WebhookAction(action) if isinstance(action, str) else action

        # Validate entity/action combination
        if not self._is_valid_combination(self._entity, self._action):
            valid_actions = VALID_ENTITY_ACTIONS.get(self._entity, set())
            raise MlflowException.invalid_parameter_value(
                f"Invalid action '{self._action}' for entity '{self._entity}'. "
                f"Valid actions are: {sorted([a.value for a in valid_actions])}"
            )

    @property
    def entity(self) -> WebhookEntity:
        return self._entity

    @property
    def action(self) -> WebhookAction:
        return self._action

    @staticmethod
    def _is_valid_combination(entity: WebhookEntity, action: WebhookAction) -> bool:
        """
        Check if an entity/action combination is valid.

        Args:
            entity: The webhook entity
            action: The webhook action

        Returns:
            True if the combination is valid, False otherwise
        """
        valid_actions = VALID_ENTITY_ACTIONS.get(entity, set())
        return action in valid_actions

    @classmethod
    def from_proto(cls, proto: ProtoWebhookEvent) -> Self:
        return cls(
            entity=WebhookEntity.from_proto(proto.entity),
            action=WebhookAction.from_proto(proto.action),
        )

    @classmethod
    def from_str(cls, event_str: WebhookEventStr) -> Self:
        """
        Create a WebhookEvent from a dot-separated string representation.

        Args:
            event_str: Valid webhook event string (e.g., "registered_model.created")

        Returns:
            A WebhookEvent instance
        """
        match event_str.split("."):
            case [entity_str, action_str]:
                try:
                    entity = WebhookEntity(entity_str)
                    action = WebhookAction(action_str)
                    return cls(entity=entity, action=action)
                except ValueError as e:
                    raise MlflowException.invalid_parameter_value(
                        f"Invalid entity or action in event string: {event_str}. Error: {e}"
                    )
            case _:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid event string format: {event_str}. "
                    "Expected format: 'entity.action' (e.g., 'registered_model.created')"
                )

    def to_proto(self) -> ProtoWebhookEvent:
        event = ProtoWebhookEvent()
        event.entity = self.entity.to_proto()
        event.action = self.action.to_proto()
        return event

    def __str__(self) -> str:
        return f"{self.entity.value}.{self.action.value}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WebhookEvent):
            return False
        return self.entity == other.entity and self.action == other.action

    def __hash__(self) -> int:
        return hash((self.entity, self.action))

    def __repr__(self) -> str:
        return f"WebhookEvent(entity={self.entity}, action={self.action})"


class Webhook:
    """
    MLflow entity for Webhook.
    """

    def __init__(
        self,
        webhook_id: str,
        name: str,
        url: str,
        events: list[WebhookEvent],
        creation_timestamp: int,
        last_updated_timestamp: int,
        description: str | None = None,
        status: str | WebhookStatus = WebhookStatus.ACTIVE,
        secret: str | None = None,
    ):
        """
        Initialize a Webhook entity.

        Args:
            webhook_id: Unique webhook identifier
            name: Human-readable webhook name
            url: Webhook endpoint URL
            events: List of WebhookEvent objects that trigger this webhook
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
        self._events = events
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
    def events(self) -> list[WebhookEvent]:
        return self._events

    @property
    def description(self) -> str | None:
        return self._description

    @property
    def status(self) -> WebhookStatus:
        return self._status

    @property
    def secret(self) -> str | None:
        return self._secret

    @property
    def creation_timestamp(self) -> int:
        return self._creation_timestamp

    @property
    def last_updated_timestamp(self) -> int:
        return self._last_updated_timestamp

    @classmethod
    def from_proto(cls, proto: ProtoWebhook) -> Self:
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
        webhook = ProtoWebhook()
        webhook.webhook_id = self.webhook_id
        webhook.name = self.name
        webhook.url = self.url
        webhook.events.extend([event.to_proto() for event in self.events])
        if self.description:
            webhook.description = self.description
        webhook.status = self.status.to_proto()
        webhook.creation_timestamp = self.creation_timestamp
        webhook.last_updated_timestamp = self.last_updated_timestamp
        return webhook

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
        response_status: int | None = None,
        response_body: str | None = None,
        error_message: str | None = None,
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
    def response_status(self) -> int | None:
        return self._response_status

    @property
    def response_body(self) -> str | None:
        return self._response_body

    @property
    def error_message(self) -> str | None:
        return self._error_message

    @classmethod
    def from_proto(cls, proto: ProtoWebhookTestResult) -> Self:
        return cls(
            success=proto.success,
            response_status=proto.response_status or None,
            response_body=proto.response_body or None,
            error_message=proto.error_message or None,
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
            f"success={self.success!r}, "
            f"response_status={self.response_status!r}, "
            f"response_body={self.response_body!r}, "
            f"error_message={self.error_message!r}"
            f")"
        )

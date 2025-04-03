import json
from enum import Enum
from typing import Optional

from mlflow.entities.model_registry._model_registry_entity import _ModelRegistryEntity
from mlflow.protos.model_registry_pb2 import Webhook as ProtoWebhook


class WebhookEventTrigger(Enum):
    TAG = "tag"
    ALIAS = "alias"


class Webhook(_ModelRegistryEntity):
    """Webhook object associated with an event."""

    def __init__(
        self,
        name: str,
        url: str,
        event_trigger: WebhookEventTrigger,
        key: Optional[str],
        value: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        payload: Optional[dict[str, str]] = None,
        description: Optional[str] = None,
        creation_timestamp: Optional[int] = None,
        last_updated_timestamp: Optional[int] = None,
    ):
        # Constructor is called only from within the system by various backend stores.
        super().__init__()
        self._name = name
        self._description = description
        self._url = url
        self._event_trigger = event_trigger
        self._key = key
        self._value = value
        self._headers = headers
        self._payload = payload
        self._creation_timestamp = creation_timestamp
        self._last_updated_timestamp = last_updated_timestamp

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def creation_timestamp(self):
        """Integer. Model version creation timestamp (milliseconds since the Unix epoch)."""
        return self._creation_timestamp

    @property
    def last_updated_timestamp(self):
        """Integer. Timestamp of last update for this model version (milliseconds since the Unix
        epoch).
        """
        return self._last_updated_timestamp

    @last_updated_timestamp.setter
    def last_updated_timestamp(self, updated_timestamp):
        self._last_updated_timestamp = updated_timestamp

    @property
    def description(self):
        """String. Description"""
        return self._description

    @description.setter
    def description(self, description):
        self._description = description

    @property
    def url(self):
        """String. url"""
        return self._url

    @url.setter
    def url(self, url):
        self._url = url

    @property
    def event_trigger(self):
        """String. event_trigger"""
        if isinstance(self._event_trigger, str):
            return self._event_trigger
        if isinstance(self._event_trigger, WebhookEventTrigger):
            return self._event_trigger.value

    @event_trigger.setter
    def event_trigger(self, event_trigger: str):
        self._event_trigger = WebhookEventTrigger(event_trigger)

    @property
    def key(self):
        """String. key"""
        return self._key

    @key.setter
    def key(self, key):
        self._key = key

    @property
    def value(self):
        """String. value"""
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def headers(self):
        """Dict. headers"""
        return self._headers

    @headers.setter
    def headers(self, headers):
        self._headers = json.loads(headers)

    @property
    def payload(self):
        """Dict. payload"""
        return self._payload

    @payload.setter
    def payload(self, payload):
        self._payload = json.loads(payload)

    @classmethod
    def from_proto(self, proto) -> dict:
        """Convert webhook to dictionary representation"""
        return self(
            name=proto.name,
            description=proto.description,
            url=proto.url,
            event_trigger=WebhookEventTrigger(proto.event_trigger),
            key=proto.key,
            value=proto.value,
            headers=proto.headers,
            payload=proto.payload,
            creation_timestamp=proto.creation_timestamp,
            last_updated_timestamp=proto.last_updated_timestamp,
        )

    def to_proto(self) -> "Webhook":
        # returns mlflow.protos.model_registry_pb2.Webhook
        webhook = ProtoWebhook()
        webhook.name = self.name
        if self.creation_timestamp is not None:
            webhook.creation_timestamp = self.creation_timestamp
        if self.last_updated_timestamp:
            webhook.last_updated_timestamp = self.last_updated_timestamp
        if self.description:
            webhook.description = self.description
        if self.url:
            webhook.url = self.url
        if self.event_trigger:
            webhook.event_trigger = self.event_trigger
        if self.key:
            webhook.key = self.key
        if self.value:
            webhook.value = self.value
        if self.headers:
            webhook.headers.update(self.headers)
        if self.payload:
            webhook.payload.update(self.payload)
        return webhook

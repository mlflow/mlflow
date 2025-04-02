from mlflow.entities.model_registry.webhook import Webhook, WebhookEventTrigger

from tests.helper_functions import random_str


def _check(
    webhook,
    name,
    creation_timestamp,
    last_updated_timestamp,
    description,
    url,
    event_trigger,
    key,
    value,
    headers,
    payload,
):
    assert isinstance(webhook, Webhook)
    assert webhook.name == name
    assert webhook.creation_timestamp == creation_timestamp
    assert webhook.last_updated_timestamp == last_updated_timestamp
    assert webhook.description == description
    assert webhook.url == url
    assert webhook.event_trigger == event_trigger.value
    assert webhook.key == key
    assert webhook.value == value
    assert webhook.headers == headers
    assert webhook.payload == payload


def test_creation_and_hydration():
    name = random_str()
    description = random_str()
    webhook_1 = Webhook(
        name=name,
        description=description,
        url="http://localhost:8080",
        event_trigger=WebhookEventTrigger.TAG,
        key="stage",
        value="staging",
        headers={"X-HEADER": "header_value"},
        payload={"custom_payload": "payload_value"},
        creation_timestamp=1,
        last_updated_timestamp=2,
    )
    _check(
        webhook_1,
        name,
        1,
        2,
        description,
        "http://localhost:8080",
        WebhookEventTrigger.TAG,
        "stage",
        "staging",
        {"X-HEADER": "header_value"},
        {"custom_payload": "payload_value"},
    )

    as_dict = {
        "name": name,
        "creation_timestamp": 1,
        "last_updated_timestamp": 2,
        "description": description,
        "url": "http://localhost:8080",
        "event_trigger": WebhookEventTrigger.TAG.value,
        "key": "stage",
        "value": "staging",
        "headers": {"X-HEADER": "header_value"},
        "payload": {"custom_payload": "payload_value"},
    }
    assert dict(webhook_1) == as_dict

    proto = webhook_1.to_proto()
    assert proto.name == name
    assert proto.creation_timestamp == 1
    assert proto.last_updated_timestamp == 2
    assert proto.description == description
    assert proto.url == "http://localhost:8080"
    assert proto.event_trigger == WebhookEventTrigger.TAG.value
    assert proto.key == "stage"
    assert proto.value == "staging"
    assert proto.headers == {"X-HEADER": "header_value"}
    assert proto.payload == {"custom_payload": "payload_value"}
    webhook_2 = Webhook.from_proto(proto)
    _check(
        webhook_2,
        name,
        1,
        2,
        description,
        "http://localhost:8080",
        WebhookEventTrigger.TAG,
        "stage",
        "staging",
        {"X-HEADER": "header_value"},
        {"custom_payload": "payload_value"},
    )

    webhook_3 = Webhook.from_dictionary(as_dict)
    _check(
        webhook_3,
        name,
        1,
        2,
        description,
        "http://localhost:8080",
        WebhookEventTrigger.TAG,
        "stage",
        "staging",
        {"X-HEADER": "header_value"},
        {"custom_payload": "payload_value"},
    )


def test_string_repr():
    webhook = Webhook(
        creation_timestamp=1000,
        description="about a webhook",
        event_trigger=WebhookEventTrigger.TAG,
        headers={"X-HEADER": "header_value"},
        key="stage",
        last_updated_timestamp=2002,
        name="webhook",
        payload={"custom_payload": "payload_value"},
        url="http://localhost:8080",
        value="staging",
    )
    assert (
        str(webhook) == "<Webhook: creation_timestamp=1000, description='about a webhook', "
        "event_trigger='tag', headers={'X-HEADER': 'header_value'}, key='stage', "
        "last_updated_timestamp=2002, name='webhook', "
        "payload={'custom_payload': 'payload_value'}, "
        "url='http://localhost:8080', value='staging'>"
    )

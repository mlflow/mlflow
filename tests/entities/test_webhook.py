import pytest

from mlflow.entities.webhook import (
    Webhook,
    WebhookAction,
    WebhookEntity,
    WebhookEvent,
    WebhookStatus,
    WebhookTestResult,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.webhooks_pb2 import WebhookAction as ProtoWebhookAction
from mlflow.protos.webhooks_pb2 import WebhookEntity as ProtoWebhookEntity
from mlflow.protos.webhooks_pb2 import WebhookStatus as ProtoWebhookStatus


@pytest.mark.parametrize(
    ("proto_status", "status_enum"),
    [
        (ProtoWebhookStatus.ACTIVE, WebhookStatus.ACTIVE),
        (ProtoWebhookStatus.DISABLED, WebhookStatus.DISABLED),
    ],
)
def test_webhook_status_proto_conversion(proto_status, status_enum):
    assert WebhookStatus.from_proto(proto_status) == status_enum
    assert status_enum.to_proto() == proto_status


@pytest.mark.parametrize(
    ("entity_enum", "proto_entity"),
    [
        (WebhookEntity.REGISTERED_MODEL, ProtoWebhookEntity.REGISTERED_MODEL),
        (WebhookEntity.MODEL_VERSION, ProtoWebhookEntity.MODEL_VERSION),
        (WebhookEntity.MODEL_VERSION_TAG, ProtoWebhookEntity.MODEL_VERSION_TAG),
        (WebhookEntity.MODEL_VERSION_ALIAS, ProtoWebhookEntity.MODEL_VERSION_ALIAS),
    ],
)
def test_webhook_entity_proto_conversion(entity_enum, proto_entity):
    assert WebhookEntity.from_proto(proto_entity) == entity_enum
    assert entity_enum.to_proto() == proto_entity


@pytest.mark.parametrize(
    ("action_enum", "proto_action"),
    [
        (WebhookAction.CREATED, ProtoWebhookAction.CREATED),
        (WebhookAction.UPDATED, ProtoWebhookAction.UPDATED),
        (WebhookAction.DELETED, ProtoWebhookAction.DELETED),
        (WebhookAction.SET, ProtoWebhookAction.SET),
    ],
)
def test_webhook_action_proto_conversion(action_enum, proto_action):
    assert WebhookAction.from_proto(proto_action) == action_enum
    assert action_enum.to_proto() == proto_action


def test_webhook_event_creation():
    event = WebhookEvent(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED)
    assert event.entity == WebhookEntity.REGISTERED_MODEL
    assert event.action == WebhookAction.CREATED


def test_webhook_event_from_string():
    event = WebhookEvent("registered_model", "created")
    assert event.entity == WebhookEntity.REGISTERED_MODEL
    assert event.action == WebhookAction.CREATED


def test_webhook_event_invalid_combination():
    with pytest.raises(
        MlflowException, match="Invalid action 'updated' for entity 'model_version_tag'"
    ):
        WebhookEvent(WebhookEntity.MODEL_VERSION_TAG, WebhookAction.UPDATED)


def test_webhook_event_from_str():
    event = WebhookEvent.from_str("registered_model.created")
    assert event.entity == WebhookEntity.REGISTERED_MODEL
    assert event.action == WebhookAction.CREATED


def test_webhook_event_from_str_invalid_format():
    with pytest.raises(MlflowException, match="Invalid event string format"):
        WebhookEvent.from_str("invalid_format")


def test_webhook_event_to_str():
    event = WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)
    assert str(event) == "model_version.created"


def test_webhook_event_proto_conversion():
    event = WebhookEvent(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED)
    proto_event = event.to_proto()
    event_from_proto = WebhookEvent.from_proto(proto_event)
    assert event_from_proto.entity == event.entity
    assert event_from_proto.action == event.action


def test_webhook_event_equality():
    event1 = WebhookEvent(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED)
    event2 = WebhookEvent(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED)
    event3 = WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)

    assert event1 == event2
    assert event1 != event3
    assert hash(event1) == hash(event2)
    assert hash(event1) != hash(event3)


def test_webhook_event_invalid_entity_action_combination():
    with pytest.raises(
        MlflowException, match="Invalid action 'deleted' for entity 'registered_model'"
    ):
        WebhookEvent(WebhookEntity.REGISTERED_MODEL, WebhookAction.DELETED)


def test_webhook_proto_conversion():
    events = [
        WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED),
        WebhookEvent(WebhookEntity.MODEL_VERSION_ALIAS, WebhookAction.CREATED),
    ]
    webhook = Webhook(
        webhook_id="webhook123",
        name="Test Webhook",
        url="https://example.com/webhook",
        events=events,
        description="Test webhook description",
        status=WebhookStatus.ACTIVE,
        secret="my-secret",
        creation_timestamp=1234567890,
        last_updated_timestamp=1234567900,
    )
    proto_webhook = webhook.to_proto()
    webhook_from_proto = Webhook.from_proto(proto_webhook)
    assert webhook_from_proto.webhook_id == webhook.webhook_id
    assert webhook_from_proto.name == webhook.name
    assert webhook_from_proto.url == webhook.url
    assert webhook_from_proto.events == webhook.events
    assert webhook_from_proto.description == webhook.description
    assert webhook_from_proto.status == webhook.status
    assert webhook_from_proto.creation_timestamp == webhook.creation_timestamp
    assert webhook_from_proto.last_updated_timestamp == webhook.last_updated_timestamp


def test_webhook_no_secret_in_repr():
    events = [WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)]
    webhook = Webhook(
        webhook_id="webhook123",
        name="Test Webhook",
        url="https://example.com/webhook",
        events=events,
        creation_timestamp=1234567890,
        last_updated_timestamp=1234567900,
        description="Test webhook description",
        status=WebhookStatus.ACTIVE,
        secret="my-secret",
    )
    assert "my-secret" not in repr(webhook)


def test_webhook_invalid_events():
    with pytest.raises(MlflowException, match="Webhook events cannot be empty"):
        Webhook(
            webhook_id="webhook123",
            name="Test Webhook",
            url="https://example.com/webhook",
            events=[],
            creation_timestamp=1234567890,
            last_updated_timestamp=1234567900,
        )


def test_webhook_test_result():
    # Test successful result
    result = WebhookTestResult(
        success=True,
        response_status=200,
        response_body='{"status": "ok"}',
    )
    assert result.success is True
    assert result.response_status == 200
    assert result.response_body == '{"status": "ok"}'
    assert result.error_message is None

    # Test failed result
    result = WebhookTestResult(
        success=False,
        response_status=500,
        error_message="Internal server error",
    )
    assert result.success is False
    assert result.response_status == 500
    assert result.error_message == "Internal server error"
    assert result.response_body is None

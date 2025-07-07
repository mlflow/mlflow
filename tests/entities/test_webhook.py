import pytest

from mlflow.entities.webhook import Webhook, WebhookEvent, WebhookStatus, WebhookTestResult
from mlflow.exceptions import MlflowException
from mlflow.protos.webhooks_pb2 import WebhookEvent as ProtoWebhookEvent
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
    ("event_enum", "proto_event"),
    [
        (WebhookEvent.REGISTERED_MODEL_CREATED, ProtoWebhookEvent.REGISTERED_MODEL_CREATED),
        (WebhookEvent.MODEL_VERSION_CREATED, ProtoWebhookEvent.MODEL_VERSION_CREATED),
        (WebhookEvent.MODEL_VERSION_TAG_SET, ProtoWebhookEvent.MODEL_VERSION_TAG_SET),
        (WebhookEvent.MODEL_VERSION_TAG_DELETED, ProtoWebhookEvent.MODEL_VERSION_TAG_DELETED),
        (WebhookEvent.MODEL_VERSION_ALIAS_CREATED, ProtoWebhookEvent.MODEL_VERSION_ALIAS_CREATED),
        (WebhookEvent.MODEL_VERSION_ALIAS_DELETED, ProtoWebhookEvent.MODEL_VERSION_ALIAS_DELETED),
    ],
)
def test_webhook_event_proto_conversion(event_enum, proto_event):
    assert WebhookEvent.from_proto(proto_event) == event_enum
    assert event_enum.to_proto() == proto_event


def test_webhook_proto_conversion():
    webhook = Webhook(
        webhook_id="webhook123",
        name="Test Webhook",
        url="https://example.com/webhook",
        events=[WebhookEvent.MODEL_VERSION_CREATED, WebhookEvent.MODEL_VERSION_ALIAS_CREATED],
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
    webhook = Webhook(
        webhook_id="webhook123",
        name="Test Webhook",
        url="https://example.com/webhook",
        events=[WebhookEvent.MODEL_VERSION_CREATED],
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
            description="Test webhook description",
            status=WebhookStatus.ACTIVE,
            creation_timestamp=1234567900,
            last_updated_timestamp=1234567900,
        )


def test_webhook_test_result_proto_conversion():
    result = WebhookTestResult(
        success=True,
        response_status=200,
        response_body='{"message": "success"}',
        error_message=None,
    )
    proto_result = result.to_proto()
    result_from_proto = WebhookTestResult.from_proto(proto_result)
    assert result_from_proto.success == result.success
    assert result_from_proto.response_status == result.response_status
    assert result_from_proto.response_body == result.response_body
    assert result_from_proto.error_message == result.error_message

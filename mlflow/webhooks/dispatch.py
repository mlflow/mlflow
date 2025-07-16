import hashlib
import hmac
import json
import logging
from typing import Optional

import requests

from mlflow.entities.webhook import WebhookEvent, WebhookTestResult
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.webhooks.constants import WEBHOOK_SIGNATURE_HEADER
from mlflow.webhooks.types import WebhookPayload

_logger = logging.getLogger(__name__)


def _generate_hmac_signature(secret: str, payload_bytes: bytes) -> str:
    """Generate HMAC-SHA256 signature for webhook payload.

    Args:
        secret: The webhook secret key
        payload_bytes: The serialized payload bytes

    Returns:
        The HMAC signature in the format "sha256=<hex_digest>"
    """
    signature = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
    return f"sha256={signature}"


def _send_webhook_request(
    url: str,
    payload: WebhookPayload,
    secret: Optional[str] = None,
) -> WebhookTestResult:
    """Send a webhook request to the specified URL.

    Args:
        url: The webhook URL to send the request to
        payload: The payload to send
        secret: Optional secret for HMAC signature

    Returns:
        WebhookTestResult indicating success/failure and response details
    """
    try:
        payload_bytes = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        # Add HMAC signature if secret is configured
        if secret:
            signature = _generate_hmac_signature(secret, payload_bytes)
            headers[WEBHOOK_SIGNATURE_HEADER] = signature

        response = requests.post(url, data=payload_bytes, headers=headers, timeout=30)

        return WebhookTestResult(
            success=response.status_code < 400,
            response_status=response.status_code,
            response_body=response.text[:1000] if response.text else None,  # Truncate response
        )
    except Exception as e:
        return WebhookTestResult(
            success=False,
            error_message=str(e)[:500],  # Truncate error message
        )


def _dispatch_webhook_impl(
    *,
    event: WebhookEvent,
    payload: WebhookPayload,
    store: AbstractStore,
) -> None:
    # TODO: Make this non-blocking
    for webhook in store.list_webhooks():
        if event in webhook.events:
            _send_webhook_request(webhook.url, payload, webhook.secret)


def dispatch_webhook(
    *,
    event: WebhookEvent,
    payload: WebhookPayload,
    store: AbstractStore,
) -> None:
    try:
        _dispatch_webhook_impl(event=event, payload=payload, store=store)
    except Exception as e:
        _logger.error(
            f"Failed to dispatch webhook for event {event}: {e}",
            exc_info=True,
        )


def test_webhook(
    webhook_id: str, store: AbstractStore, event: Optional[WebhookEvent] = None
) -> WebhookTestResult:
    """Test a webhook by sending a test payload.

    Args:
        webhook_id: The ID of the webhook to test
        store: The model registry store to retrieve webhook details
        event: Optional event type to test. If not specified, uses the first event from webhook.

    Returns:
        WebhookTestResult indicating success/failure and response details
    """
    try:
        webhook = store.get_webhook(webhook_id)

        # Use provided event or the first event type for testing
        test_event = event or webhook.events[0]

        # Generate example payload based on the event type
        if test_event == WebhookEvent.REGISTERED_MODEL_CREATED:
            from mlflow.webhooks.types import RegisteredModelCreatedPayload

            test_payload = RegisteredModelCreatedPayload.example()
        elif test_event == WebhookEvent.MODEL_VERSION_CREATED:
            from mlflow.webhooks.types import ModelVersionCreatedPayload

            test_payload = ModelVersionCreatedPayload.example()
        elif test_event == WebhookEvent.MODEL_VERSION_TAG_SET:
            from mlflow.webhooks.types import ModelVersionTagSetPayload

            test_payload = ModelVersionTagSetPayload.example()
        elif test_event == WebhookEvent.MODEL_VERSION_TAG_DELETED:
            from mlflow.webhooks.types import ModelVersionTagDeletedPayload

            test_payload = ModelVersionTagDeletedPayload.example()
        elif test_event == WebhookEvent.MODEL_VERSION_ALIAS_CREATED:
            from mlflow.webhooks.types import ModelVersionAliasCreatedPayload

            test_payload = ModelVersionAliasCreatedPayload.example()
        elif test_event == WebhookEvent.MODEL_VERSION_ALIAS_DELETED:
            from mlflow.webhooks.types import ModelVersionAliasDeletedPayload

            test_payload = ModelVersionAliasDeletedPayload.example()
        else:
            # Default to MODEL_VERSION_CREATED payload
            from mlflow.webhooks.types import ModelVersionCreatedPayload

            test_payload = ModelVersionCreatedPayload.example()

        return _send_webhook_request(webhook.url, test_payload, webhook.secret)
    except Exception as e:
        return WebhookTestResult(
            success=False,
            error_message=f"Failed to test webhook: {str(e)[:500]}",
        )

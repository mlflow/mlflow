import hashlib
import hmac
import json
import logging
from typing import Optional

import requests

from mlflow.entities.webhook import Webhook, WebhookEvent, WebhookTestResult
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.webhooks.constants import WEBHOOK_SIGNATURE_HEADER
from mlflow.webhooks.types import WebhookPayload, get_example_payload_for_event

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
    webhook: Webhook,
    payload: WebhookPayload,
) -> requests.Response:
    """Send a webhook request to the specified URL.

    Args:
        webhook: The webhook object containing the URL and secret
        payload: The payload to send

    Returns:
        requests.Response object from the webhook request
    """
    payload_bytes = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    # Add HMAC signature if secret is configured
    if webhook.secret:
        signature = _generate_hmac_signature(webhook.secret, payload_bytes)
        headers[WEBHOOK_SIGNATURE_HEADER] = signature

    return requests.post(webhook.url, data=payload_bytes, headers=headers, timeout=30)


def _dispatch_webhook_impl(
    *,
    event: WebhookEvent,
    payload: WebhookPayload,
    store: AbstractStore,
) -> None:
    # TODO: Make this non-blocking
    for webhook in store.list_webhooks():
        if event in webhook.events:
            try:
                _send_webhook_request(webhook, payload)
            except Exception as e:
                _logger.error(
                    f"Failed to send webhook to {webhook.url} for event {event}: {e}",
                    exc_info=True,
                )


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


def test_webhook(webhook: Webhook, event: Optional[WebhookEvent] = None) -> WebhookTestResult:
    """Test a webhook by sending a test payload.

    Args:
        webhook: The webhook object to test
        event: Optional event type to test. If not specified, uses the first event from webhook.

    Returns:
        WebhookTestResult indicating success/failure and response details
    """
    # Use provided event or the first event type for testing
    test_event = event or webhook.events[0]
    try:
        test_payload = get_example_payload_for_event(test_event)
        response = _send_webhook_request(webhook=webhook, payload=test_payload)
        return WebhookTestResult(
            success=response.status_code < 400,
            response_status=response.status_code,
            response_body=response.text,
        )
    except Exception as e:
        return WebhookTestResult(
            success=False,
            error_message=f"Failed to test webhook: {e!r}",
        )

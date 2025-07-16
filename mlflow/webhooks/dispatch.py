import hashlib
import hmac
import json
import logging

import requests

from mlflow.entities.webhook import WebhookEvent
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


def _dispatch_webhook_impl(
    *,
    event: WebhookEvent,
    payload: WebhookPayload,
    store: AbstractStore,
) -> None:
    # TODO: Make this non-blocking
    for webhook in store.list_webhooks():
        if event in webhook.events:
            payload_bytes = json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            # Add HMAC signature if secret is configured
            if webhook.secret:
                signature = _generate_hmac_signature(webhook.secret, payload_bytes)
                headers[WEBHOOK_SIGNATURE_HEADER] = signature

            requests.post(webhook.url, data=payload_bytes, headers=headers)


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

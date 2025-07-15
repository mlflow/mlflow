import logging

import requests

from mlflow.entities.webhook import WebhookEvent
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.webhooks.types import WebhookPayload

_logger = logging.getLogger(__name__)


def _dispatch_webhook_impl(
    *,
    event: WebhookEvent,
    payload: WebhookPayload,
    store: AbstractStore,
) -> None:
    # TODO: Make this non-blocking
    for webhook in store.list_webhooks():
        if event in webhook.events:
            # TODO: Implement HMAC signature verification
            requests.post(webhook.url, json=payload)


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

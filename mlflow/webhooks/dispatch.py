from typing import Any

import requests

from mlflow.entities.webhook import WebhookEvent
from mlflow.store.model_registry.abstract_store import AbstractStore


def dispatch_webhook(
    *,
    event: WebhookEvent,
    payload: dict[str, Any],
    store: AbstractStore,
) -> None:
    # TODO: Make this non-blocking
    for webhook in store.list_webhooks():
        if event in webhook.events:
            # TODO: Implement HMAC signature verification
            requests.post(webhook.url, json=payload)

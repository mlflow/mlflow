"""Webhook dispatch implementation following Standard Webhooks conventions.

This module implements webhook delivery patterns similar to the Standard Webhooks
specification (https://www.standardwebhooks.com), providing consistent and secure
webhook delivery with HMAC signature verification and timestamp-based replay protection.
"""

import base64
import hashlib
import hmac
import json
import logging
import random
import time
import uuid
from datetime import datetime, timezone

import requests

from mlflow.entities.webhook import Webhook, WebhookEvent, WebhookTestResult
from mlflow.environment_variables import (
    MLFLOW_WEBHOOK_REQUEST_MAX_RETRIES,
    MLFLOW_WEBHOOK_REQUEST_TIMEOUT,
)
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.webhooks.constants import (
    WEBHOOK_DELIVERY_ID_HEADER,
    WEBHOOK_SIGNATURE_HEADER,
    WEBHOOK_SIGNATURE_VERSION,
    WEBHOOK_TIMESTAMP_HEADER,
)
from mlflow.webhooks.types import (
    WebhookPayload,
    get_example_payload_for_event,
)

_logger = logging.getLogger(__name__)


def _calculate_backoff_with_jitter(attempt: int, jitter_fraction: float = 0.1) -> float:
    """Calculate exponential backoff time with jitter.

    Args:
        attempt: The retry attempt number (0-based)
        jitter_fraction: Fraction of base backoff to use as maximum jitter (default: 0.1)

    Returns:
        Backoff time in seconds with jitter applied
    """
    base_backoff = 2**attempt  # Exponential backoff: 1s, 2s, 4s, 8s...
    jitter = random.uniform(0, base_backoff * jitter_fraction)
    return base_backoff + jitter


def _generate_hmac_signature(secret: str, delivery_id: str, timestamp: str, payload: str) -> str:
    """Generate webhook HMAC-SHA256 signature.

    Args:
        secret: The webhook secret key
        delivery_id: The unique delivery ID
        timestamp: Unix timestamp as string
        payload: The JSON payload as string

    Returns:
        The signature in the format "v1,<base64_encoded_signature>"
    """
    # Signature format: delivery_id.timestamp.payload
    signed_content = f"{delivery_id}.{timestamp}.{payload}"
    signature = hmac.new(
        secret.encode("utf-8"), signed_content.encode("utf-8"), hashlib.sha256
    ).digest()
    signature_b64 = base64.b64encode(signature).decode("utf-8")
    return f"{WEBHOOK_SIGNATURE_VERSION},{signature_b64}"


def _send_webhook_request(
    webhook: Webhook,
    payload: WebhookPayload,
    event: WebhookEvent,
) -> requests.Response:
    """Send a webhook request to the specified URL with retry logic.

    Args:
        webhook: The webhook object containing the URL and secret
        payload: The payload to send
        event: The webhook event type

    Returns:
        requests.Response object from the webhook request
    """
    # Create webhook payload with metadata
    webhook_payload = {
        "type": event.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": payload,
    }

    payload_json = json.dumps(webhook_payload)
    payload_bytes = payload_json.encode("utf-8")

    # Generate IDs and timestamps for webhooks
    delivery_id = str(uuid.uuid4())
    unix_timestamp = str(int(time.time()))

    # MLflow webhook headers
    headers = {
        "Content-Type": "application/json",
        WEBHOOK_DELIVERY_ID_HEADER: delivery_id,
        WEBHOOK_TIMESTAMP_HEADER: unix_timestamp,
    }

    # Add signature if secret is configured
    if webhook.secret:
        signature = _generate_hmac_signature(
            webhook.secret, delivery_id, unix_timestamp, payload_json
        )
        headers[WEBHOOK_SIGNATURE_HEADER] = signature

    timeout = MLFLOW_WEBHOOK_REQUEST_TIMEOUT.get()
    max_retries = MLFLOW_WEBHOOK_REQUEST_MAX_RETRIES.get()

    # Retry logic with exponential backoff and jitter
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                webhook.url, data=payload_bytes, headers=headers, timeout=timeout
            )

            should_retry = response.status_code >= 500 or response.status_code == 429
            if not should_retry:
                # Return successful response or non-retryable errors
                return response

            if attempt < max_retries:
                backoff_time = _calculate_backoff_with_jitter(attempt)

                # For 429, check if there's a Retry-After header
                if response.status_code == 429 and (
                    retry_after := response.headers.get("Retry-After")
                ):
                    try:
                        retry_after = int(retry_after)
                        backoff_time = max(backoff_time, retry_after)
                    except (ValueError, TypeError):
                        # If Retry-After is not a valid integer, use calculated backoff
                        pass

                _logger.warning(
                    f"Webhook request to {webhook.url} failed with status {response.status_code}. "
                    f"Retrying in {backoff_time:.2f}s (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(backoff_time)
            else:
                # Last attempt failed, return the response
                return response

        except requests.RequestException as e:
            if attempt < max_retries:
                backoff_time = _calculate_backoff_with_jitter(attempt)

                _logger.warning(
                    f"Webhook request to {webhook.url} failed: {e}. "
                    f"Retrying in {backoff_time:.2f}s (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(backoff_time)
            else:
                # Last attempt failed, re-raise the exception
                raise

    # This should never be reached, but included for completeness
    raise RuntimeError("Unexpected error in webhook retry logic")


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
                _send_webhook_request(webhook, payload, event)
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


def test_webhook(webhook: Webhook, event: WebhookEvent | None = None) -> WebhookTestResult:
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
        response = _send_webhook_request(webhook=webhook, payload=test_payload, event=test_event)
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

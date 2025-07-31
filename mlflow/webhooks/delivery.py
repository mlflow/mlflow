"""Webhook delivery implementation following Standard Webhooks conventions.

This module implements webhook delivery patterns similar to the Standard Webhooks
specification (https://www.standardwebhooks.com), providing consistent and secure
webhook delivery with HMAC signature verification and timestamp-based replay protection.
"""

import base64
import hashlib
import hmac
import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests
import urllib3
from packaging.version import Version
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from mlflow.entities.webhook import Webhook, WebhookEvent, WebhookTestResult
from mlflow.environment_variables import (
    MLFLOW_WEBHOOK_DELIVERY_MAX_WORKERS,
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

# Thread pool for non-blocking webhook delivery
_webhook_delivery_executor = ThreadPoolExecutor(
    max_workers=MLFLOW_WEBHOOK_DELIVERY_MAX_WORKERS.get(),
    thread_name_prefix="webhook-delivery",
)

# Shared session for webhook requests (thread-safe)
_webhook_session: requests.Session | None = None
_webhook_session_lock: threading.Lock = threading.Lock()


def _create_webhook_session() -> requests.Session:
    """Create a new webhook session with retry configuration.

    Returns:
        Configured requests.Session object
    """
    max_retries = MLFLOW_WEBHOOK_REQUEST_MAX_RETRIES.get()

    # urllib3 >= 2.0 supports additional features
    extra_kwargs = {}
    if Version(urllib3.__version__) >= Version("2.0"):
        extra_kwargs["backoff_jitter"] = 1.0  # Add up to 1 second of jitter

    retry_strategy = Retry(
        total=max_retries,
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
        allowed_methods=["POST"],  # Only retry POST requests
        backoff_factor=1.0,  # Exponential backoff: 1s, 2s, 4s, etc.
        backoff_max=60.0,  # Cap maximum backoff at 60 seconds
        respect_retry_after_header=True,  # Automatically handle Retry-After headers
        raise_on_status=False,  # Don't raise on these status codes
        **extra_kwargs,
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def _get_or_create_webhook_session() -> requests.Session:
    """Get or create a shared webhook session with retry configuration.

    Returns:
        Configured requests.Session object
    """
    global _webhook_session

    if _webhook_session is None:  # To avoid unnecessary locking
        with _webhook_session_lock:
            if _webhook_session is None:
                _webhook_session = _create_webhook_session()

    return _webhook_session


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
    session: requests.Session,
) -> requests.Response:
    """Send a webhook request to the specified URL with retry logic.

    Args:
        webhook: The webhook object containing the URL and secret
        payload: The payload to send
        event: The webhook event type
        session: Configured requests session with retry logic

    Returns:
        requests.Response object from the webhook request
    """
    # Create webhook payload with metadata
    webhook_payload = {
        "entity": event.entity.value,
        "action": event.action.value,
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

    try:
        return session.post(webhook.url, data=payload_bytes, headers=headers, timeout=timeout)
    except requests.exceptions.RetryError as e:
        # urllib3 exhausted all retries
        max_retries = MLFLOW_WEBHOOK_REQUEST_MAX_RETRIES.get()
        _logger.error(f"Webhook request to {webhook.url} failed after {max_retries} retries: {e}")
        raise
    except requests.RequestException as e:
        # Other request errors
        _logger.error(f"Webhook request to {webhook.url} failed: {e}")
        raise


def _send_webhook_with_error_handling(
    webhook: Webhook,
    payload: WebhookPayload,
    event: WebhookEvent,
    session: requests.Session,
) -> None:
    try:
        _send_webhook_request(webhook, payload, event, session)
    except Exception as e:
        _logger.error(
            f"Failed to send webhook to {webhook.url} for event {event}: {e}",
            exc_info=True,
        )


def _deliver_webhook_impl(
    *,
    event: WebhookEvent,
    payload: WebhookPayload,
    store: AbstractStore,
) -> None:
    session = _get_or_create_webhook_session()
    for webhook in store.list_webhooks():
        if webhook.status.is_active() and event in webhook.events:
            _webhook_delivery_executor.submit(
                _send_webhook_with_error_handling,
                webhook,
                payload,
                event,
                session,
            )


def deliver_webhook(
    *,
    event: WebhookEvent,
    payload: WebhookPayload,
    store: AbstractStore,
) -> None:
    try:
        _deliver_webhook_impl(event=event, payload=payload, store=store)
    except Exception as e:
        _logger.error(
            f"Failed to deliver webhook for event {event}: {e}",
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
    session = _get_or_create_webhook_session()
    try:
        test_payload = get_example_payload_for_event(test_event)
        response = _send_webhook_request(
            webhook=webhook, payload=test_payload, event=test_event, session=session
        )
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

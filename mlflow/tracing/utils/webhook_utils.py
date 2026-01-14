"""
Webhook utility module for MLflow tracing.

Currently, webhook triggering for tracing events is **not implemented client-side**.
This file defines a stub function to preserve interface compatibility for future backend integration.
"""

import logging

_logger = logging.getLogger(__name__)

def fire_tracing_webhook(event_type: str, trace, metadata: dict | None = None):
    """
    Stub for future backend-triggered tracing webhooks.

    Args:
        event_type (str): The type of tracing event (e.g., latency, error).
        trace: The trace object containing trace details.
        metadata (dict | None): Optional metadata related to the event.
    """
    _logger.debug(
        f"[TracingWebhook] Event '{event_type}' requested for trace {getattr(trace, 'id', None)}, "
        "but client-side webhook triggering is disabled. "
        "This will be handled by the backend in future versions."
    )

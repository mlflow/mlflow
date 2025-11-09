import time
import requests
from mlflow.tracking import MlflowClient

def fire_tracing_webhook(event_type: str, trace, metadata: dict | None = None):
    """
    Send tracing-related webhook events to registered URLs.
    Uses existing MLflow webhooks if configured via MlflowClient().
    """
    client = MlflowClient()
    webhooks = client.list_webhooks()

    for hook in webhooks:
        if hook.event != event_type:
            continue

        payload = {
            "event_type": event_type,
            "trace_id": getattr(trace, "id", None),
            "timestamp": time.time(),
            "metadata": metadata or {},
        }

        headers = {}
        if getattr(hook, "http_headers", None):
            headers.update(hook.http_headers)

        try:
            requests.post(hook.url, json=payload, headers=headers, timeout=5)
        except Exception as e:
            print(f"[TracingWebhook] Failed to send to {hook.url}: {e}")
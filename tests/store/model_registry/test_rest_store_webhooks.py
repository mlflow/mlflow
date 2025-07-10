import subprocess
import sys
from pathlib import Path
from typing import Generator

import pytest

from mlflow.entities.webhook import WebhookEvent, WebhookStatus
from mlflow.exceptions import MlflowException
from mlflow.store.model_registry.rest_store import RestStore
from mlflow.utils.rest_utils import MlflowHostCreds

from tests.helper_functions import get_safe_port


@pytest.fixture
def server(tmp_path: Path) -> Generator[str, None, None]:
    port = get_safe_port()
    sqlite_db_path = tmp_path / "mlflow.db"

    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--dev",
            f"--port={port}",
            "--backend-store-uri",
            f"sqlite:///{sqlite_db_path}",
        ],
        cwd=tmp_path,
    ) as process:
        url = f"http://localhost:{port}"
        try:
            yield url
        finally:
            process.terminate()


@pytest.fixture
def store(server: str) -> RestStore:
    return RestStore(lambda: MlflowHostCreds(server))


def test_create_webhook(store: RestStore):
    webhook = store.create_webhook(
        name="test_webhook",
        url="https://example.com/webhook",
        events=[WebhookEvent.MODEL_VERSION_CREATED],
    )
    assert webhook.name == "test_webhook"
    assert webhook.url == "https://example.com/webhook"
    assert webhook.secret is None
    assert webhook.events == [WebhookEvent.MODEL_VERSION_CREATED]

    webhook = store.get_webhook(webhook.webhook_id)
    assert webhook.name == "test_webhook"
    assert webhook.url == "https://example.com/webhook"
    assert webhook.secret is None

    # With secret
    webhook_with_secret = store.create_webhook(
        name="test_webhook_with_secret",
        url="https://example.com/webhook_with_secret",
        events=[WebhookEvent.MODEL_VERSION_CREATED],
        secret="my_secret",
    )
    assert webhook_with_secret.name == "test_webhook_with_secret"
    assert webhook_with_secret.url == "https://example.com/webhook_with_secret"
    assert webhook_with_secret.secret == "my_secret"
    assert webhook_with_secret.events == [WebhookEvent.MODEL_VERSION_CREATED]

    # Multiple events
    webhook_multiple_events = store.create_webhook(
        name="test_webhook_multiple_events",
        url="https://example.com/webhook_multiple_events",
        events=[
            WebhookEvent.MODEL_VERSION_CREATED,
            WebhookEvent.MODEL_VERSION_TRANSITIONED_STAGE,
        ],
    )
    assert webhook_multiple_events.name == "test_webhook_multiple_events"
    assert webhook_multiple_events.url == "https://example.com/webhook_multiple_events"
    assert webhook_multiple_events.events == [
        WebhookEvent.MODEL_VERSION_CREATED,
        WebhookEvent.MODEL_VERSION_TRANSITIONED_STAGE,
    ]


@pytest.mark.parametrize(
    ("invalid_url", "expected_match"),
    [
        ("", r"Webhook URL cannot be empty or just whitespace"),
        ("   ", r"Webhook URL cannot be empty or just whitespace"),
        ("ftp://example.com", r"Invalid webhook URL scheme"),
        ("http://[invalid", r"Invalid webhook URL"),
    ],
)
def test_update_webhook_invalid_urls(store, invalid_url, expected_match):
    # Create a valid webhook first
    webhook = store.create_webhook(
        name="test_webhook",
        url="https://example.com/webhook",
        events=[WebhookEvent.MODEL_VERSION_CREATED],
    )
    with pytest.raises(MlflowException, match=expected_match):
        store.update_webhook(webhook_id=webhook.webhook_id, url=invalid_url)


def test_get_webhook(store: RestStore):
    events = [WebhookEvent.MODEL_VERSION_CREATED]
    created_webhook = store.create_webhook(
        name="test_webhook", url="https://example.com/webhook", events=events
    )
    retrieved_webhook = store.get_webhook(created_webhook.webhook_id)
    assert retrieved_webhook.webhook_id == created_webhook.webhook_id
    assert retrieved_webhook.name == "test_webhook"
    assert retrieved_webhook.url == "https://example.com/webhook"
    assert retrieved_webhook.events == events


def test_get_webhook_not_found(store: RestStore):
    with pytest.raises(MlflowException, match="Webhook with ID nonexistent not found"):
        store.get_webhook("nonexistent")


def test_list_webhooks(store: RestStore):
    # Create multiple webhooks
    webhook1 = store.create_webhook(
        name="webhook1", url="https://example.com/1", events=[WebhookEvent.MODEL_VERSION_CREATED]
    )
    webhook2 = store.create_webhook(
        name="webhook2", url="https://example.com/2", events=[WebhookEvent.REGISTERED_MODEL_CREATED]
    )
    webhooks, token = store.list_webhooks()
    assert len(webhooks) == 2
    assert token is None
    webhook_ids = {w.webhook_id for w in webhooks}
    assert webhook1.webhook_id in webhook_ids
    assert webhook2.webhook_id in webhook_ids


def test_update_webhook(store: RestStore):
    events = [WebhookEvent.MODEL_VERSION_CREATED]
    webhook = store.create_webhook(
        name="original_name", url="https://example.com/original", events=events
    )
    # Update webhook
    new_events = [WebhookEvent.MODEL_VERSION_CREATED, WebhookEvent.REGISTERED_MODEL_CREATED]
    updated_webhook = store.update_webhook(
        webhook_id=webhook.webhook_id,
        name="updated_name",
        url="https://example.com/updated",
        events=new_events,
        description="Updated description",
        secret="new_secret",
        status=WebhookStatus.DISABLED,
    )
    assert updated_webhook.webhook_id == webhook.webhook_id
    assert updated_webhook.name == "updated_name"
    assert updated_webhook.url == "https://example.com/updated"
    assert updated_webhook.events == new_events
    assert updated_webhook.description == "Updated description"
    assert updated_webhook.status == WebhookStatus.DISABLED
    assert updated_webhook.last_updated_timestamp > webhook.last_updated_timestamp


def test_update_webhook_partial(store: RestStore):
    events = [WebhookEvent.MODEL_VERSION_CREATED]
    webhook = store.create_webhook(
        name="original_name",
        url="https://example.com/original",
        events=events,
        description="Original description",
    )
    # Update only the name
    updated_webhook = store.update_webhook(
        webhook_id=webhook.webhook_id,
        name="updated_name",
    )
    assert updated_webhook.name == "updated_name"
    assert updated_webhook.url == "https://example.com/original"
    assert updated_webhook.events == events
    assert updated_webhook.description == "Original description"


def test_update_webhook_not_found(store: RestStore):
    with pytest.raises(MlflowException, match="Webhook with ID nonexistent not found"):
        store.update_webhook(webhook_id="nonexistent", name="new_name")


def test_delete_webhook(store: RestStore):
    events = [WebhookEvent.MODEL_VERSION_CREATED]
    webhook = store.create_webhook(
        name="test_webhook",
        url="https://example.com/webhook",
        events=events,
    )
    store.delete_webhook(webhook.webhook_id)
    with pytest.raises(MlflowException, match=r"Webhook with ID .* not found"):
        store.get_webhook(webhook.webhook_id)
    webhooks, _ = store.list_webhooks()
    webhook_ids = {w.webhook_id for w in webhooks}
    assert webhook.webhook_id not in webhook_ids


def test_delete_webhook_not_found(store: RestStore):
    with pytest.raises(MlflowException, match="Webhook with ID nonexistent not found"):
        store.delete_webhook("nonexistent")

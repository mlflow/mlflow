"""
This test file verifies webhook CRUD operations with the REST client,
testing both server handlers and the REST client together.
"""

from pathlib import Path
from typing import Iterator

import pytest
from cryptography.fernet import Fernet

from mlflow.entities.webhook import WebhookAction, WebhookEntity, WebhookEvent, WebhookStatus
from mlflow.environment_variables import MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY
from mlflow.exceptions import MlflowException
from mlflow.server import handlers
from mlflow.server.fastapi_app import app
from mlflow.server.handlers import initialize_backend_stores
from mlflow.store.model_registry.rest_store import RestStore
from mlflow.utils.rest_utils import MlflowHostCreds

from tests.helper_functions import get_safe_port
from tests.tracking.integration_test_utils import ServerThread


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[RestStore]:
    """Set up a local MLflow server with proper webhook encryption key support."""
    # Set up encryption key for webhooks using monkeypatch
    encryption_key = Fernet.generate_key().decode("utf-8")
    monkeypatch.setenv(MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY.name, encryption_key)

    # Configure backend stores
    backend_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    default_artifact_root = tmp_path.as_uri()

    # Force-reset backend stores before each test
    handlers._tracking_store = None
    handlers._model_registry_store = None
    initialize_backend_stores(backend_uri, default_artifact_root=default_artifact_root)

    # Start server and return RestStore
    with ServerThread(app, get_safe_port()) as url:
        yield RestStore(lambda: MlflowHostCreds(url))


def test_create_webhook(store: RestStore):
    webhook = store.create_webhook(
        name="test_webhook",
        url="https://example.com/webhook",
        events=[WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)],
    )
    assert webhook.name == "test_webhook"
    assert webhook.url == "https://example.com/webhook"
    assert webhook.secret is None
    assert webhook.events == [WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)]

    webhook = store.get_webhook(webhook.webhook_id)
    assert webhook.name == "test_webhook"
    assert webhook.url == "https://example.com/webhook"
    assert webhook.secret is None

    # With secret
    webhook_with_secret = store.create_webhook(
        name="test_webhook_with_secret",
        url="https://example.com/webhook_with_secret",
        events=[WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)],
        secret="my_secret",
    )
    assert webhook_with_secret.name == "test_webhook_with_secret"
    assert webhook_with_secret.url == "https://example.com/webhook_with_secret"
    assert webhook_with_secret.secret is None
    assert webhook_with_secret.events == [
        WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)
    ]

    # Multiple events
    webhook_multiple_events = store.create_webhook(
        name="test_webhook_multiple_events",
        url="https://example.com/webhook_multiple_events",
        events=[
            WebhookEvent(WebhookEntity.MODEL_VERSION_ALIAS, WebhookAction.CREATED),
            WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED),
        ],
    )
    assert webhook_multiple_events.name == "test_webhook_multiple_events"
    assert webhook_multiple_events.url == "https://example.com/webhook_multiple_events"
    assert sorted(
        webhook_multiple_events.events, key=lambda e: (e.entity.value, e.action.value)
    ) == [
        WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED),
        WebhookEvent(WebhookEntity.MODEL_VERSION_ALIAS, WebhookAction.CREATED),
    ]
    assert webhook_multiple_events.secret is None


def test_get_webhook(store: RestStore):
    events = [WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)]
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
    # Create more webhooks than max_results
    created_webhooks = []
    for i in range(5):
        webhook = store.create_webhook(
            name=f"webhook{i}",
            url=f"https://example.com/{i}",
            events=[WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)],
        )
        created_webhooks.append(webhook)
    # Test pagination with max_results=2
    webhooks_page = store.list_webhooks(max_results=2)
    assert len(webhooks_page) == 2
    assert webhooks_page.token is not None
    # Get next page
    next_webhooks_page = store.list_webhooks(max_results=2, page_token=webhooks_page.token)
    assert len(next_webhooks_page) == 2
    assert next_webhooks_page.token is not None
    # Verify we don't get duplicates
    first_page_ids = {w.webhook_id for w in webhooks_page}
    second_page_ids = {w.webhook_id for w in next_webhooks_page}
    assert first_page_ids.isdisjoint(second_page_ids)


def test_update_webhook(store: RestStore):
    events = [WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)]
    webhook = store.create_webhook(
        name="original_name", url="https://example.com/original", events=events
    )
    # Update webhook
    new_events = [
        WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED),
        WebhookEvent(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED),
    ]
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
    events = [WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)]
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


@pytest.mark.parametrize(
    ("invalid_url", "expected_match"),
    [
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
        events=[WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)],
    )
    with pytest.raises(MlflowException, match=expected_match):
        store.update_webhook(webhook_id=webhook.webhook_id, url=invalid_url)


def test_delete_webhook(store: RestStore):
    events = [WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)]
    webhook = store.create_webhook(
        name="test_webhook",
        url="https://example.com/webhook",
        events=events,
    )
    store.delete_webhook(webhook.webhook_id)
    with pytest.raises(MlflowException, match=r"Webhook with ID .* not found"):
        store.get_webhook(webhook.webhook_id)
    webhooks_page = store.list_webhooks()
    webhook_ids = {w.webhook_id for w in webhooks_page}
    assert webhook.webhook_id not in webhook_ids


def test_delete_webhook_not_found(store: RestStore):
    with pytest.raises(MlflowException, match="Webhook with ID nonexistent not found"):
        store.delete_webhook("nonexistent")

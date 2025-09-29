"""Tests for webhook delivery functionality."""

import tempfile
from unittest.mock import Mock, patch

import pytest

from mlflow.entities.webhook import WebhookAction, WebhookEntity, WebhookEvent
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.store.tracking.file_store import FileStore
from mlflow.webhooks.delivery import deliver_webhook


@pytest.fixture
def file_store():
    """Create a FileStore instance for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield FileStore(tmp_dir)


@pytest.fixture
def mock_abstract_store():
    """Create a mock AbstractStore that is not a FileStore."""
    return Mock(spec=AbstractStore)


@pytest.fixture
def webhook_event():
    """Create a webhook event for testing."""
    return WebhookEvent(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED)


@pytest.fixture
def webhook_payload():
    """Create a webhook payload for testing."""
    return {"name": "test_model", "description": "Test model"}


def test_deliver_webhook_exits_early_for_file_store(file_store, webhook_event, webhook_payload):
    """Test that deliver_webhook exits early when provided with a FileStore."""
    with patch("mlflow.webhooks.delivery._deliver_webhook_impl") as mock_impl:
        deliver_webhook(
            event=webhook_event,
            payload=webhook_payload,
            store=file_store,
        )

        # _deliver_webhook_impl should not be called for FileStore
        mock_impl.assert_not_called()


def test_deliver_webhook_calls_impl_for_non_file_store(
    mock_abstract_store, webhook_event, webhook_payload
):
    """Test that deliver_webhook calls implementation for non-FileStore instances."""
    with patch("mlflow.webhooks.delivery._deliver_webhook_impl") as mock_impl:
        deliver_webhook(
            event=webhook_event,
            payload=webhook_payload,
            store=mock_abstract_store,
        )

        # _deliver_webhook_impl should be called for non-FileStore
        mock_impl.assert_called_once_with(
            event=webhook_event,
            payload=webhook_payload,
            store=mock_abstract_store,
        )


def test_deliver_webhook_handles_exception_for_non_file_store(
    mock_abstract_store, webhook_event, webhook_payload
):
    """Test that deliver_webhook handles exceptions properly for non-FileStore instances."""
    with patch("mlflow.webhooks.delivery._deliver_webhook_impl") as mock_impl:
        with patch("mlflow.webhooks.delivery._logger") as mock_logger:
            # Make _deliver_webhook_impl raise an exception
            mock_impl.side_effect = Exception("Test exception")

            # This should not raise an exception
            deliver_webhook(
                event=webhook_event,
                payload=webhook_payload,
                store=mock_abstract_store,
            )

            # Verify that the error was logged
            mock_logger.error.assert_called_once()
            assert "Failed to deliver webhook for event" in str(mock_logger.error.call_args)


def test_deliver_webhook_no_exception_for_file_store(file_store, webhook_event, webhook_payload):
    """Test that deliver_webhook does not raise exceptions for FileStore, even with errors."""
    with patch("mlflow.webhooks.delivery._deliver_webhook_impl") as mock_impl:
        with patch("mlflow.webhooks.delivery._logger") as mock_logger:
            # Make _deliver_webhook_impl raise an exception (though it shouldn't be called)
            mock_impl.side_effect = Exception("Test exception")

            # This should not raise an exception and should return early
            deliver_webhook(
                event=webhook_event,
                payload=webhook_payload,
                store=file_store,
            )

            # _deliver_webhook_impl should not be called, so no error should be logged
            mock_impl.assert_not_called()
            mock_logger.error.assert_not_called()

from pathlib import Path
from unittest.mock import patch

import pytest

from mlflow.entities.webhook import WebhookAction, WebhookEntity, WebhookEvent
from mlflow.store.model_registry.file_store import FileStore
from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore
from mlflow.webhooks.delivery import deliver_webhook


@pytest.fixture
def file_store(tmp_path: Path) -> FileStore:
    return FileStore(str(tmp_path))


@pytest.fixture
def sql_store(tmp_path: Path) -> SqlAlchemyStore:
    db_file = tmp_path / "test.db"
    db_uri = f"sqlite:///{db_file}"
    return SqlAlchemyStore(db_uri)


@pytest.fixture
def webhook_event() -> WebhookEvent:
    return WebhookEvent(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED)


@pytest.fixture
def webhook_payload() -> dict[str, str]:
    return {"name": "test_model", "description": "Test model"}


def test_deliver_webhook_exits_early_for_file_store(
    file_store: FileStore, webhook_event: WebhookEvent, webhook_payload: dict[str, str]
) -> None:
    with patch("mlflow.webhooks.delivery._deliver_webhook_impl") as mock_impl:
        deliver_webhook(
            event=webhook_event,
            payload=webhook_payload,
            store=file_store,
        )

        # _deliver_webhook_impl should not be called for FileStore
        mock_impl.assert_not_called()


def test_deliver_webhook_calls_impl_for_sql_store(
    sql_store: SqlAlchemyStore, webhook_event: WebhookEvent, webhook_payload: dict[str, str]
) -> None:
    with patch("mlflow.webhooks.delivery._deliver_webhook_impl") as mock_impl:
        deliver_webhook(
            event=webhook_event,
            payload=webhook_payload,
            store=sql_store,
        )

        # _deliver_webhook_impl should be called for SqlAlchemyStore
        mock_impl.assert_called_once_with(
            event=webhook_event,
            payload=webhook_payload,
            store=sql_store,
        )


def test_deliver_webhook_handles_exception_for_sql_store(
    sql_store: SqlAlchemyStore, webhook_event: WebhookEvent, webhook_payload: dict[str, str]
) -> None:
    with (
        patch("mlflow.webhooks.delivery._deliver_webhook_impl", side_effect=Exception("Test")),
        patch("mlflow.webhooks.delivery._logger") as mock_logger,
    ):
        # This should not raise an exception
        deliver_webhook(
            event=webhook_event,
            payload=webhook_payload,
            store=sql_store,
        )

        # Verify that the error was logged
        mock_logger.error.assert_called_once()
        assert "Failed to deliver webhook for event" in str(mock_logger.error.call_args)


def test_deliver_webhook_no_exception_for_file_store(
    file_store: FileStore, webhook_event: WebhookEvent, webhook_payload: dict[str, str]
) -> None:
    with (
        patch(
            "mlflow.webhooks.delivery._deliver_webhook_impl", side_effect=Exception("Test")
        ) as mock_impl,
        patch("mlflow.webhooks.delivery._logger") as mock_logger,
    ):
        # This should not raise an exception and should return early
        deliver_webhook(
            event=webhook_event,
            payload=webhook_payload,
            store=file_store,
        )

        # _deliver_webhook_impl should not be called, so no error should be logged
        mock_impl.assert_not_called()
        mock_logger.error.assert_not_called()

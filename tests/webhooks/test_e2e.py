import contextlib
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import psutil
import pytest
import requests
from cryptography.fernet import Fernet

from mlflow import MlflowClient
from mlflow.entities.webhook import WebhookAction, WebhookEntity, WebhookEvent

from tests.helper_functions import get_safe_port
from tests.webhooks.app import WEBHOOK_SECRET


@dataclass
class WebhookLogEntry:
    endpoint: str
    headers: dict[str, str]
    status_code: int
    payload: dict[str, Any]
    error: str | None = None
    attempt: int | None = None


def wait_until_ready(health_endpoint: str, max_attempts: int = 10) -> None:
    for _ in range(max_attempts):
        try:
            resp = requests.get(health_endpoint, timeout=2)
            if resp.status_code == 200:
                return
        except requests.RequestException:
            time.sleep(1)
    raise RuntimeError(f"Failed to start server at {health_endpoint}")


@contextlib.contextmanager
def _run_mlflow_server(tmp_path: Path) -> Generator[str, None, None]:
    port = get_safe_port()
    backend_store_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_root = (tmp_path / "artifacts").as_uri()
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            f"--port={port}",
            f"--backend-store-uri={backend_store_uri}",
            f"--default-artifact-root={artifact_root}",
        ],
        cwd=tmp_path,
        env=(
            os.environ.copy()
            | {
                "MLFLOW_WEBHOOK_ALLOWED_SCHEMES": "http",
                "MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY": Fernet.generate_key().decode(),
                "MLFLOW_WEBHOOK_REQUEST_MAX_RETRIES": "3",
                "MLFLOW_WEBHOOK_REQUEST_TIMEOUT": "10",
                "MLFLOW_WEBHOOK_CACHE_TTL": "0",  # Disable caching for tests
            }
        ),
    ) as prc:
        try:
            url = f"http://localhost:{port}"
            wait_until_ready(f"{url}/health")
            yield url
        finally:
            # Kill the gunicorn processes spawned by mlflow server
            try:
                proc = psutil.Process(prc.pid)
            except psutil.NoSuchProcess:
                # Handle case where the process did not start correctly
                pass
            else:
                for child in proc.children(recursive=True):
                    child.terminate()

            # Kill the mlflow server process
            prc.terminate()


class AppClient:
    def __init__(self, base: str) -> None:
        self._base = base

    def get_url(self, endpoint: str) -> str:
        return f"{self._base}{endpoint}"

    def reset(self) -> None:
        """Reset both logs and counters"""
        resp = requests.post(self.get_url("/reset"))
        resp.raise_for_status()

    def get_logs(self) -> list[WebhookLogEntry]:
        response = requests.get(self.get_url("/logs"))
        response.raise_for_status()
        logs_data = response.json().get("logs", [])
        return [WebhookLogEntry(**log_data) for log_data in logs_data]

    def wait_for_logs(self, expected_count: int, timeout: float = 5.0) -> list[WebhookLogEntry]:
        """Wait for webhooks to be delivered with a timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            logs = self.get_logs()
            if len(logs) >= expected_count:
                return logs
            time.sleep(0.1)
        # Raise timeout error if expected count not reached
        logs = self.get_logs()
        raise TimeoutError(
            f"Timeout waiting for {expected_count} webhook logs. "
            f"Got {len(logs)} logs after {timeout}s timeout."
        )


@contextlib.contextmanager
def _run_app(tmp_path: Path) -> Generator[AppClient, None, None]:
    port = get_safe_port()
    app_path = Path(__file__).parent / "app.py"
    with subprocess.Popen(
        [
            sys.executable,
            app_path,
            str(port),
        ],
        cwd=tmp_path,
    ) as prc:
        try:
            url = f"http://localhost:{port}"
            wait_until_ready(f"{url}/health")
            yield AppClient(url)
        finally:
            prc.terminate()


@pytest.fixture(scope="module")
def app_client(tmp_path_factory: pytest.TempPathFactory) -> Generator[AppClient, None, None]:
    tmp_path = tmp_path_factory.mktemp("app")
    with _run_app(tmp_path) as client:
        yield client


@pytest.fixture(scope="module")
def mlflow_server(
    app_client: AppClient, tmp_path_factory: pytest.TempPathFactory
) -> Generator[str, None, None]:
    tmp_path = tmp_path_factory.mktemp("mlflow_server")
    with _run_mlflow_server(tmp_path) as url:
        yield url


@pytest.fixture(scope="module")
def mlflow_client(mlflow_server: str) -> MlflowClient:
    with pytest.MonkeyPatch.context() as mp:
        # Disable retries to fail fast
        mp.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
        return MlflowClient(tracking_uri=mlflow_server, registry_uri=mlflow_server)


@pytest.fixture(autouse=True)
def cleanup(mlflow_client: MlflowClient, app_client: AppClient) -> Generator[None, None, None]:
    yield

    for webhook in mlflow_client.list_webhooks():
        mlflow_client.delete_webhook(webhook.webhook_id)

    app_client.reset()


def test_registered_model_created(mlflow_client: MlflowClient, app_client: AppClient) -> None:
    mlflow_client.create_webhook(
        name="registered_model_created",
        url=app_client.get_url("/insecure-webhook"),
        events=[WebhookEvent(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED)],
    )
    registered_model = mlflow_client.create_registered_model(
        name="test_name",
        description="test_description",
        tags={"test_tag_key": "test_tag_value"},
    )
    logs = app_client.wait_for_logs(expected_count=1)
    assert len(logs) == 1
    assert logs[0].endpoint == "/insecure-webhook"
    assert logs[0].payload == {
        "name": registered_model.name,
        "description": registered_model.description,
        "tags": registered_model.tags,
    }


def test_model_version_created(mlflow_client: MlflowClient, app_client: AppClient) -> None:
    mlflow_client.create_webhook(
        name="model_version_created",
        url=app_client.get_url("/insecure-webhook"),
        events=[WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)],
    )
    registered_model = mlflow_client.create_registered_model(name="model_version_created")
    model_version = mlflow_client.create_model_version(
        name=registered_model.name,
        source="s3://bucket/path/to/model",
        run_id="1234567890abcdef",
        tags={"test_tag_key": "test_tag_value"},
        description="test_description",
    )
    logs = app_client.wait_for_logs(expected_count=1)
    assert len(logs) == 1
    assert logs[0].endpoint == "/insecure-webhook"
    assert logs[0].payload == {
        "name": registered_model.name,
        "version": model_version.version,
        "source": "s3://bucket/path/to/model",
        "run_id": "1234567890abcdef",
        "description": "test_description",
        "tags": {"test_tag_key": "test_tag_value"},
    }


def test_model_version_tag_set(mlflow_client: MlflowClient, app_client: AppClient) -> None:
    mlflow_client.create_webhook(
        name="model_version_tag_set",
        url=app_client.get_url("/insecure-webhook"),
        events=[WebhookEvent(WebhookEntity.MODEL_VERSION_TAG, WebhookAction.SET)],
    )
    registered_model = mlflow_client.create_registered_model(name="model_version_tag_set")
    model_version = mlflow_client.create_model_version(
        name=registered_model.name,
        source="s3://bucket/path/to/model",
        run_id="1234567890abcdef",
    )
    mlflow_client.set_model_version_tag(
        name=model_version.name,
        version=model_version.version,
        key="test_tag_key",
        value="new_value",
    )
    logs = app_client.wait_for_logs(expected_count=1)
    assert len(logs) == 1
    assert logs[0].endpoint == "/insecure-webhook"
    assert logs[0].payload == {
        "name": "model_version_tag_set",
        "version": model_version.version,
        "key": "test_tag_key",
        "value": "new_value",
    }


def test_model_version_tag_deleted(mlflow_client: MlflowClient, app_client: AppClient) -> None:
    mlflow_client.create_webhook(
        name="model_version_tag_deleted",
        url=app_client.get_url("/insecure-webhook"),
        events=[WebhookEvent(WebhookEntity.MODEL_VERSION_TAG, WebhookAction.DELETED)],
    )
    registered_model = mlflow_client.create_registered_model(name="model_version_tag_deleted")
    model_version = mlflow_client.create_model_version(
        name=registered_model.name,
        source="s3://bucket/path/to/model",
        run_id="1234567890abcdef",
        tags={"test_tag_key": "test_tag_value"},
    )
    mlflow_client.set_model_version_tag(
        name=model_version.name,
        version=model_version.version,
        key="test_tag_key",
        value="new_value",
    )
    mlflow_client.delete_model_version_tag(
        name=model_version.name, version=model_version.version, key="test_tag_key"
    )
    logs = app_client.wait_for_logs(expected_count=1)
    assert len(logs) == 1
    assert logs[0].endpoint == "/insecure-webhook"
    assert logs[0].payload == {
        "name": registered_model.name,
        "version": model_version.version,
        "key": "test_tag_key",
    }


def test_model_version_alias_created(mlflow_client: MlflowClient, app_client: AppClient) -> None:
    mlflow_client.create_webhook(
        name="model_version_alias_created",
        url=app_client.get_url("/insecure-webhook"),
        events=[WebhookEvent(WebhookEntity.MODEL_VERSION_ALIAS, WebhookAction.CREATED)],
    )
    registered_model = mlflow_client.create_registered_model(name="model_version_alias_created")
    model_version = mlflow_client.create_model_version(
        name=registered_model.name,
        source="s3://bucket/path/to/model",
        run_id="1234567890abcdef",
        tags={"test_tag_key": "test_tag_value"},
        description="test_description",
    )
    mlflow_client.set_registered_model_alias(
        name=model_version.name, version=model_version.version, alias="test_alias"
    )
    logs = app_client.wait_for_logs(expected_count=1)
    assert len(logs) == 1
    assert logs[0].endpoint == "/insecure-webhook"
    assert logs[0].payload == {
        "name": registered_model.name,
        "version": model_version.version,
        "alias": "test_alias",
    }


def test_model_version_alias_deleted(mlflow_client: MlflowClient, app_client: AppClient) -> None:
    mlflow_client.create_webhook(
        name="model_version_alias_deleted",
        url=app_client.get_url("/insecure-webhook"),
        events=[WebhookEvent(WebhookEntity.MODEL_VERSION_ALIAS, WebhookAction.DELETED)],
    )
    registered_model = mlflow_client.create_registered_model(name="model_version_alias_deleted")
    model_version = mlflow_client.create_model_version(
        name=registered_model.name,
        source="s3://bucket/path/to/model",
        run_id="1234567890abcdef",
        tags={"test_tag_key": "test_tag_value"},
        description="test_description",
    )
    mlflow_client.set_registered_model_alias(
        name=model_version.name, version=model_version.version, alias="test_alias"
    )
    mlflow_client.delete_registered_model_alias(name=model_version.name, alias="test_alias")
    logs = app_client.wait_for_logs(expected_count=1)
    assert len(logs) == 1
    assert logs[0].endpoint == "/insecure-webhook"
    assert logs[0].payload == {
        "name": registered_model.name,
        "alias": "test_alias",
    }


def test_webhook_with_secret(mlflow_client: MlflowClient, app_client: AppClient) -> None:
    # Create webhook with secret that matches the one in app.py
    mlflow_client.create_webhook(
        name="secure_webhook",
        url=app_client.get_url("/secure-webhook"),
        events=[WebhookEvent(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED)],
        secret=WEBHOOK_SECRET,
    )

    registered_model = mlflow_client.create_registered_model(
        name="test_hmac_model",
        description="Testing HMAC signature",
        tags={"env": "test"},
    )

    logs = app_client.wait_for_logs(expected_count=1)
    assert len(logs) == 1
    assert logs[0].endpoint == "/secure-webhook"
    assert logs[0].payload == {
        "name": registered_model.name,
        "description": registered_model.description,
        "tags": registered_model.tags,
    }
    assert logs[0].status_code == 200
    # HTTP headers are case-insensitive and FastAPI normalizes them to lowercase
    assert "x-mlflow-signature" in logs[0].headers
    assert logs[0].headers["x-mlflow-signature"].startswith("v1,")


def test_webhook_with_wrong_secret(mlflow_client: MlflowClient, app_client: AppClient) -> None:
    # Create webhook with wrong secret that doesn't match the one in app.py
    mlflow_client.create_webhook(
        name="wrong_secret_webhook",
        url=app_client.get_url("/secure-webhook"),
        events=[WebhookEvent(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED)],
        secret="wrong-secret",  # This doesn't match WEBHOOK_SECRET in app.py
    )

    # This should fail at the webhook endpoint due to signature mismatch
    # But MLflow will still create the registered model
    mlflow_client.create_registered_model(
        name="test_wrong_hmac",
        description="Testing wrong HMAC signature",
    )

    # The webhook request should have failed, but error should be logged
    logs = app_client.wait_for_logs(expected_count=1)
    assert len(logs) == 1
    assert logs[0].endpoint == "/secure-webhook"
    assert logs[0].error == "Invalid signature"
    assert logs[0].status_code == 401


def test_webhook_without_secret_to_secure_endpoint(
    mlflow_client: MlflowClient, app_client: AppClient
) -> None:
    # Create webhook without secret pointing to secure endpoint
    mlflow_client.create_webhook(
        name="no_secret_to_secure",
        url=app_client.get_url("/secure-webhook"),
        events=[WebhookEvent(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED)],
        # No secret provided
    )

    mlflow_client.create_registered_model(
        name="test_no_secret_to_secure",
        description="Testing no secret to secure endpoint",
    )

    # The webhook request should fail due to missing signature, but error should be logged
    logs = app_client.wait_for_logs(expected_count=1)
    assert len(logs) == 1
    assert logs[0].endpoint == "/secure-webhook"
    assert logs[0].error == "Missing signature header"
    assert logs[0].status_code == 400


def test_webhook_test_insecure_endpoint(mlflow_client: MlflowClient, app_client: AppClient) -> None:
    # Create webhook for testing
    webhook = mlflow_client.create_webhook(
        name="test_webhook",
        url=app_client.get_url("/insecure-webhook"),
        events=[WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)],
    )

    # Test the webhook
    result = mlflow_client.test_webhook(webhook.webhook_id)

    # Check that the test was successful
    assert result.success is True
    assert result.response_status == 200
    assert result.error_message is None

    # Check that the test payload was received
    logs = app_client.wait_for_logs(expected_count=1)
    assert len(logs) == 1
    assert logs[0].endpoint == "/insecure-webhook"
    assert logs[0].payload == {
        "name": "example_model",
        "version": "1",
        "source": "models:/123",
        "run_id": "abcd1234abcd5678",
        "tags": {"example_key": "example_value"},
        "description": "An example model version",
    }


def test_webhook_test_secure_endpoint(mlflow_client: MlflowClient, app_client: AppClient) -> None:
    # Create webhook with secret for testing
    webhook = mlflow_client.create_webhook(
        name="test_secure_webhook",
        url=app_client.get_url("/secure-webhook"),
        events=[WebhookEvent(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED)],
        secret=WEBHOOK_SECRET,
    )

    # Test the webhook
    result = mlflow_client.test_webhook(webhook.webhook_id)

    # Check that the test was successful
    assert result.success is True
    assert result.response_status == 200
    assert result.error_message is None

    # Check that the test payload was received with proper signature
    logs = app_client.wait_for_logs(expected_count=1)
    assert len(logs) == 1
    assert logs[0].endpoint == "/secure-webhook"
    assert logs[0].payload == {
        "name": "example_model",
        "tags": {"example_key": "example_value"},
        "description": "An example registered model",
    }

    assert logs[0].status_code == 200
    assert "x-mlflow-signature" in logs[0].headers
    assert logs[0].headers["x-mlflow-signature"].startswith("v1,")


def test_webhook_test_with_specific_event(
    mlflow_client: MlflowClient, app_client: AppClient
) -> None:
    # Create webhook that supports multiple events
    webhook = mlflow_client.create_webhook(
        name="multi_event_webhook",
        url=app_client.get_url("/insecure-webhook"),
        events=[
            WebhookEvent(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED),
            WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED),
            WebhookEvent(WebhookEntity.MODEL_VERSION_TAG, WebhookAction.SET),
        ],
    )

    # Test with a specific event (not the first one)
    result = mlflow_client.test_webhook(
        webhook.webhook_id, event=WebhookEvent(WebhookEntity.MODEL_VERSION_TAG, WebhookAction.SET)
    )

    # Check that the test was successful
    assert result.success is True
    assert result.response_status == 200
    assert result.error_message is None

    # Check that the correct payload was sent
    logs = app_client.wait_for_logs(expected_count=1)
    assert len(logs) == 1
    assert logs[0].endpoint == "/insecure-webhook"
    assert logs[0].payload == {
        "name": "example_model",
        "version": "1",
        "key": "example_key",
        "value": "example_value",
    }


def test_webhook_test_failed_endpoint(mlflow_client: MlflowClient, app_client: AppClient) -> None:
    # Create webhook pointing to non-existent endpoint
    webhook = mlflow_client.create_webhook(
        name="failed_webhook",
        url=app_client.get_url("/nonexistent-endpoint"),
        events=[WebhookEvent(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED)],
    )

    # Test the webhook
    result = mlflow_client.test_webhook(webhook.webhook_id)

    # Check that the test failed
    assert result.success is False
    assert result.response_status == 404
    assert result.error_message is None  # No error message for HTTP errors
    assert result.response_body is not None  # Should contain error response


def test_webhook_test_with_wrong_secret(mlflow_client: MlflowClient, app_client: AppClient) -> None:
    # Create webhook with wrong secret
    webhook = mlflow_client.create_webhook(
        name="wrong_secret_test_webhook",
        url=app_client.get_url("/secure-webhook"),
        events=[WebhookEvent(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED)],
        secret="wrong-secret",
    )

    # Test the webhook
    result = mlflow_client.test_webhook(webhook.webhook_id)

    # Check that the test failed due to wrong signature
    assert result.success is False
    assert result.response_status == 401
    assert result.error_message is None

    # Check that error was logged
    logs = app_client.wait_for_logs(expected_count=1)
    assert len(logs) == 1
    assert logs[0].endpoint == "/secure-webhook"
    assert logs[0].error == "Invalid signature"
    assert logs[0].status_code == 401


def test_webhook_retry_on_5xx_error(mlflow_client: MlflowClient, app_client: AppClient) -> None:
    """Test that webhooks retry on 5xx errors"""
    # Create webhook pointing to flaky endpoint
    mlflow_client.create_webhook(
        name="retry_test_webhook",
        url=app_client.get_url("/flaky-webhook"),
        events=[WebhookEvent(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED)],
    )

    # Create a registered model to trigger the webhook
    registered_model = mlflow_client.create_registered_model(
        name="test_retry_model",
        description="Testing retry logic",
    )

    logs = app_client.wait_for_logs(expected_count=3, timeout=15)

    # First two attempts should fail with 500
    assert logs[0].endpoint == "/flaky-webhook"
    assert logs[0].status_code == 500
    assert logs[0].error == "Server error (will retry)"
    assert logs[0].payload["name"] == registered_model.name

    assert logs[1].endpoint == "/flaky-webhook"
    assert logs[1].status_code == 500
    assert logs[1].error == "Server error (will retry)"

    # Third attempt should succeed
    assert logs[2].endpoint == "/flaky-webhook"
    assert logs[2].status_code == 200
    assert logs[2].error is None
    assert logs[2].payload["name"] == registered_model.name


def test_webhook_retry_on_429_rate_limit(
    mlflow_client: MlflowClient, app_client: AppClient
) -> None:
    """Test that webhooks retry on 429 rate limit errors and respect Retry-After header"""
    # Create webhook pointing to rate-limited endpoint
    mlflow_client.create_webhook(
        name="rate_limit_test_webhook",
        url=app_client.get_url("/rate-limited-webhook"),
        events=[WebhookEvent(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED)],
    )

    # Create a registered model to trigger the webhook
    registered_model = mlflow_client.create_registered_model(
        name="test_rate_limit_model",
        description="Testing 429 retry logic",
    )

    logs = app_client.wait_for_logs(expected_count=2, timeout=10)

    # First attempt should fail with 429
    assert logs[0].endpoint == "/rate-limited-webhook"
    assert logs[0].status_code == 429
    assert logs[0].error == "Rate limited"
    assert logs[0].payload["name"] == registered_model.name
    assert logs[0].attempt == 1

    # Second attempt should succeed
    assert logs[1].endpoint == "/rate-limited-webhook"
    assert logs[1].status_code == 200
    assert logs[1].error is None
    assert logs[1].payload["name"] == registered_model.name
    assert logs[1].attempt == 2

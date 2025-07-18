import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Generator

import psutil
import pytest
import requests
from cryptography.fernet import Fernet

from mlflow import MlflowClient
from mlflow.entities.webhook import WebhookEvent

from tests.helper_functions import get_safe_port
from tests.webhooks.app import WEBHOOK_SECRET


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

    def clear_logs(self) -> None:
        resp = requests.delete(self.get_url("/logs"))
        resp.raise_for_status()

    def get_logs(self) -> list[dict[str, Any]]:
        response = requests.get(self.get_url("/logs"))
        response.raise_for_status()
        return response.json().get("logs", [])


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

    app_client.clear_logs()


def test_registered_model_created(mlflow_client: MlflowClient, app_client: AppClient) -> None:
    mlflow_client.create_webhook(
        name="registered_model_created",
        url=app_client.get_url("/insecure-webhook"),
        events=[WebhookEvent.REGISTERED_MODEL_CREATED],
    )
    registered_model = mlflow_client.create_registered_model(
        name="test_name",
        description="test_description",
        tags={"test_tag_key": "test_tag_value"},
    )
    logs = app_client.get_logs()
    assert len(logs) == 1
    assert logs[0]["endpoint"] == "/insecure-webhook"
    assert logs[0]["payload"] == {
        "name": registered_model.name,
        "description": registered_model.description,
        "tags": registered_model.tags,
    }


def test_model_version_created(mlflow_client: MlflowClient, app_client: AppClient) -> None:
    mlflow_client.create_webhook(
        name="model_version_created",
        url=app_client.get_url("/insecure-webhook"),
        events=[WebhookEvent.MODEL_VERSION_CREATED],
    )
    registered_model = mlflow_client.create_registered_model(name="model_version_created")
    model_version = mlflow_client.create_model_version(
        name=registered_model.name,
        source="s3://bucket/path/to/model",
        run_id="1234567890abcdef",
        tags={"test_tag_key": "test_tag_value"},
        description="test_description",
    )
    logs = app_client.get_logs()
    assert len(logs) == 1
    assert logs[0]["endpoint"] == "/insecure-webhook"
    assert logs[0]["payload"] == {
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
        events=[WebhookEvent.MODEL_VERSION_TAG_SET],
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
    logs = app_client.get_logs()
    assert len(logs) == 1
    assert logs[0]["endpoint"] == "/insecure-webhook"
    assert logs[0]["payload"] == {
        "name": "model_version_tag_set",
        "version": model_version.version,
        "key": "test_tag_key",
        "value": "new_value",
    }


def test_model_version_tag_deleted(mlflow_client: MlflowClient, app_client: AppClient) -> None:
    mlflow_client.create_webhook(
        name="model_version_tag_deleted",
        url=app_client.get_url("/insecure-webhook"),
        events=[WebhookEvent.MODEL_VERSION_TAG_DELETED],
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
    logs = app_client.get_logs()
    assert len(logs) == 1
    assert logs[0]["endpoint"] == "/insecure-webhook"
    assert logs[0]["payload"] == {
        "name": registered_model.name,
        "version": model_version.version,
        "key": "test_tag_key",
    }


def test_model_version_alias_created(mlflow_client: MlflowClient, app_client: AppClient) -> None:
    mlflow_client.create_webhook(
        name="model_version_alias_created",
        url=app_client.get_url("/insecure-webhook"),
        events=[WebhookEvent.MODEL_VERSION_ALIAS_CREATED],
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
    logs = app_client.get_logs()
    assert len(logs) == 1
    assert logs[0]["endpoint"] == "/insecure-webhook"
    assert logs[0]["payload"] == {
        "name": registered_model.name,
        "version": model_version.version,
        "alias": "test_alias",
    }


def test_model_version_alias_deleted(mlflow_client: MlflowClient, app_client: AppClient) -> None:
    mlflow_client.create_webhook(
        name="model_version_alias_deleted",
        url=app_client.get_url("/insecure-webhook"),
        events=[WebhookEvent.MODEL_VERSION_ALIAS_DELETED],
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
    logs = app_client.get_logs()
    assert len(logs) == 1
    assert logs[0]["endpoint"] == "/insecure-webhook"
    assert logs[0]["payload"] == {
        "name": registered_model.name,
        "alias": "test_alias",
    }


def test_webhook_with_secret(mlflow_client: MlflowClient, app_client: AppClient) -> None:
    # Create webhook with secret that matches the one in app.py
    mlflow_client.create_webhook(
        name="secure_webhook",
        url=app_client.get_url("/secure-webhook"),
        events=[WebhookEvent.REGISTERED_MODEL_CREATED],
        secret=WEBHOOK_SECRET,
    )

    registered_model = mlflow_client.create_registered_model(
        name="test_hmac_model",
        description="Testing HMAC signature",
        tags={"env": "test"},
    )

    logs = app_client.get_logs()
    assert len(logs) == 1
    assert logs[0]["endpoint"] == "/secure-webhook"
    assert logs[0]["payload"] == {
        "name": registered_model.name,
        "description": registered_model.description,
        "tags": registered_model.tags,
    }
    assert logs[0]["status_code"] == 200
    # HTTP headers are case-insensitive and FastAPI normalizes them to lowercase
    assert "x-mlflow-signature" in logs[0]["headers"]
    assert logs[0]["headers"]["x-mlflow-signature"].startswith("sha256=")


def test_webhook_with_wrong_secret(mlflow_client: MlflowClient, app_client: AppClient) -> None:
    # Create webhook with wrong secret that doesn't match the one in app.py
    mlflow_client.create_webhook(
        name="wrong_secret_webhook",
        url=app_client.get_url("/secure-webhook"),
        events=[WebhookEvent.REGISTERED_MODEL_CREATED],
        secret="wrong-secret",  # This doesn't match WEBHOOK_SECRET in app.py
    )

    # This should fail at the webhook endpoint due to signature mismatch
    # But MLflow will still create the registered model
    mlflow_client.create_registered_model(
        name="test_wrong_hmac",
        description="Testing wrong HMAC signature",
    )

    # The webhook request should have failed, but error should be logged
    logs = app_client.get_logs()
    assert len(logs) == 1
    assert logs[0]["endpoint"] == "/secure-webhook"
    assert logs[0]["error"] == "Invalid signature"
    assert logs[0]["status_code"] == 401


def test_webhook_without_secret_to_secure_endpoint(
    mlflow_client: MlflowClient, app_client: AppClient
) -> None:
    # Create webhook without secret pointing to secure endpoint
    mlflow_client.create_webhook(
        name="no_secret_to_secure",
        url=app_client.get_url("/secure-webhook"),
        events=[WebhookEvent.REGISTERED_MODEL_CREATED],
        # No secret provided
    )

    mlflow_client.create_registered_model(
        name="test_no_secret_to_secure",
        description="Testing no secret to secure endpoint",
    )

    # The webhook request should fail due to missing signature, but error should be logged
    logs = app_client.get_logs()
    assert len(logs) == 1
    assert logs[0]["endpoint"] == "/secure-webhook"
    assert logs[0]["error"] == "Missing signature header"
    assert logs[0]["status_code"] == 400


def test_webhook_test_insecure_endpoint(mlflow_client: MlflowClient, app_client: AppClient) -> None:
    # Create webhook for testing
    webhook = mlflow_client.create_webhook(
        name="test_webhook",
        url=app_client.get_url("/insecure-webhook"),
        events=[WebhookEvent.MODEL_VERSION_CREATED],
    )

    # Test the webhook
    result = mlflow_client.test_webhook(webhook.webhook_id)

    # Check that the test was successful
    assert result.success is True
    assert result.response_status == 200
    assert result.error_message is None

    # Check that the test payload was received
    logs = app_client.get_logs()
    assert len(logs) == 1
    assert logs[0]["endpoint"] == "/insecure-webhook"
    assert logs[0]["payload"] == {
        "name": "example_model",
        "version": "1",
        "source": "runs:/abcd1234abcd5678/model",
        "run_id": "abcd1234abcd5678",
        "tags": {"example_key": "example_value"},
        "description": "An example model version",
    }


def test_webhook_test_secure_endpoint(mlflow_client: MlflowClient, app_client: AppClient) -> None:
    # Create webhook with secret for testing
    webhook = mlflow_client.create_webhook(
        name="test_secure_webhook",
        url=app_client.get_url("/secure-webhook"),
        events=[WebhookEvent.REGISTERED_MODEL_CREATED],
        secret=WEBHOOK_SECRET,
    )

    # Test the webhook
    result = mlflow_client.test_webhook(webhook.webhook_id)

    # Check that the test was successful
    assert result.success is True
    assert result.response_status == 200
    assert result.error_message is None

    # Check that the test payload was received with proper signature
    logs = app_client.get_logs()
    assert len(logs) == 1
    assert logs[0]["endpoint"] == "/secure-webhook"
    assert logs[0]["payload"] == {
        "name": "example_model",
        "tags": {"example_key": "example_value"},
        "description": "An example registered model",
    }
    assert logs[0]["status_code"] == 200
    assert "x-mlflow-signature" in logs[0]["headers"]
    assert logs[0]["headers"]["x-mlflow-signature"].startswith("sha256=")


def test_webhook_test_with_specific_event(
    mlflow_client: MlflowClient, app_client: AppClient
) -> None:
    # Create webhook that supports multiple events
    webhook = mlflow_client.create_webhook(
        name="multi_event_webhook",
        url=app_client.get_url("/insecure-webhook"),
        events=[
            WebhookEvent.REGISTERED_MODEL_CREATED,
            WebhookEvent.MODEL_VERSION_CREATED,
            WebhookEvent.MODEL_VERSION_TAG_SET,
        ],
    )

    # Test with a specific event (not the first one)
    result = mlflow_client.test_webhook(
        webhook.webhook_id, event=WebhookEvent.MODEL_VERSION_TAG_SET
    )

    # Check that the test was successful
    assert result.success is True
    assert result.response_status == 200
    assert result.error_message is None

    # Check that the correct payload was sent
    logs = app_client.get_logs()
    assert len(logs) == 1
    assert logs[0]["endpoint"] == "/insecure-webhook"
    assert logs[0]["payload"] == {
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
        events=[WebhookEvent.REGISTERED_MODEL_CREATED],
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
        events=[WebhookEvent.REGISTERED_MODEL_CREATED],
        secret="wrong-secret",
    )

    # Test the webhook
    result = mlflow_client.test_webhook(webhook.webhook_id)

    # Check that the test failed due to wrong signature
    assert result.success is False
    assert result.response_status == 401
    assert result.error_message is None

    # Check that error was logged
    logs = app_client.get_logs()
    assert len(logs) == 1
    assert logs[0]["endpoint"] == "/secure-webhook"
    assert logs[0]["error"] == "Invalid signature"
    assert logs[0]["status_code"] == 401

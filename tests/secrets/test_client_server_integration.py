import hashlib
import os
from unittest import mock

import pytest
from cryptography.fernet import Fernet

from mlflow.exceptions import MlflowException
from mlflow.secrets.client import SecretsClient
from mlflow.secrets.crypto import SecretManager
from mlflow.secrets.scope import SecretScope
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore


@pytest.fixture
def master_key():
    return Fernet.generate_key().decode()


@pytest.fixture
def store(master_key, tmp_path):
    db_uri = f"sqlite:///{tmp_path}/test.db"
    artifact_root = f"file://{tmp_path}/artifacts"
    os.environ["MLFLOW_SECRET_MASTER_KEY"] = master_key
    os.environ["MLFLOW_SECRETS_VALIDATION_MODE"] = "skip"
    store = SqlAlchemyStore(db_uri, artifact_root)
    yield store
    os.environ.pop("MLFLOW_SECRET_MASTER_KEY", None)
    os.environ.pop("MLFLOW_SECRETS_VALIDATION_MODE", None)


@pytest.fixture
def client(master_key):
    os.environ["MLFLOW_SECRET_MASTER_KEY"] = master_key
    client = SecretsClient(tracking_uri="http://localhost:5000")
    yield client
    os.environ.pop("MLFLOW_SECRET_MASTER_KEY", None)


def test_client_encrypts_before_transmission(client):
    with mock.patch("mlflow.utils.rest_utils.http_request") as mock_request:
        mock_request.return_value = {"success": True}

        client.set_secret("api_key", "secret_value_123", SecretScope.GLOBAL)

        assert mock_request.called
        call_kwargs = mock_request.call_args[1]
        request_data = call_kwargs["json"]

        assert "encrypted_name" in request_data
        assert "encrypted_value" in request_data
        assert "encrypted_dek" in request_data
        assert "integrity_hash" in request_data

        assert request_data["encrypted_name"] != "api_key"
        assert request_data["encrypted_value"] != "secret_value_123"


def test_integrity_hash_validation():
    from mlflow.server.handlers_secrets import _compute_integrity_hash

    name = "test_name"
    value = "test_value"

    client_hash = hashlib.sha256(f"{name}:{value}".encode()).hexdigest()
    server_hash = _compute_integrity_hash(name, value)

    assert client_hash == server_hash


def test_integrity_hash_mismatch_raises_error():
    secret_manager = SecretManager()
    dek = secret_manager.generate_dek()
    encrypted_name = secret_manager.encrypt_with_dek("test_name", dek)
    encrypted_value = secret_manager.encrypt_with_dek("test_value", dek)
    encrypted_dek = secret_manager.encrypt_dek(dek)

    wrong_hash = hashlib.sha256(b"wrong:data").hexdigest()

    from mlflow.server import app

    with app.test_client() as client:
        with mock.patch("mlflow.server.handlers._get_tracking_store") as mock_get_store:
            mock_store = mock.MagicMock()
            mock_get_store.return_value = mock_store

            response = client.post(
                "/api/3.0/mlflow/secrets/create",
                json={
                    "encrypted_name": encrypted_name,
                    "encrypted_value": encrypted_value,
                    "encrypted_dek": encrypted_dek,
                    "scope": SecretScope.GLOBAL.value,
                    "integrity_hash": wrong_hash,
                },
            )

            assert response.status_code in [400, 500]


def test_list_secrets_returns_only_names(client):
    with mock.patch("mlflow.utils.rest_utils.http_request") as mock_request:
        mock_request.return_value = {"secret_names": ["key1", "key2", "key3"]}

        names = client.list_secret_names(SecretScope.GLOBAL)

        assert names == ["key1", "key2", "key3"]
        assert mock_request.called

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["method"] == "GET"
        assert "list" in call_kwargs["endpoint"]


def test_delete_secret(client):
    with mock.patch("mlflow.utils.rest_utils.http_request") as mock_request:
        mock_request.return_value = {"success": True}

        client.delete_secret("old_key", SecretScope.GLOBAL)

        assert mock_request.called
        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["method"] == "DELETE"
        assert call_kwargs["json"]["name"] == "old_key"


def test_end_to_end_secret_flow(store):
    secret_manager = SecretManager()
    dek = secret_manager.generate_dek()

    name = "production_api_key"
    value = "sk-1234567890abcdef"

    encrypted_name = secret_manager.encrypt_with_dek(name, dek)
    encrypted_value = secret_manager.encrypt_with_dek(value, dek)

    decrypted_name = secret_manager.decrypt_with_dek(encrypted_name, dek)
    decrypted_value = secret_manager.decrypt_with_dek(encrypted_value, dek)

    assert decrypted_name == name
    assert decrypted_value == value

    integrity_hash = hashlib.sha256(f"{name}:{value}".encode()).hexdigest()
    recomputed_hash = hashlib.sha256(f"{decrypted_name}:{decrypted_value}".encode()).hexdigest()
    assert integrity_hash == recomputed_hash

    store.set_secret(name=name, value=value, scope=SecretScope.GLOBAL.value)

    retrieved_value = store.get_secret(name=name, scope=SecretScope.GLOBAL.value)
    assert retrieved_value == value

    names = store.list_secret_names(scope=SecretScope.GLOBAL.value)
    assert name in names

    store.delete_secret(name=name, scope=SecretScope.GLOBAL.value)
    names_after_delete = store.list_secret_names(scope=SecretScope.GLOBAL.value)
    assert name not in names_after_delete


def test_scorer_scope_with_scope_id(store):
    store.set_secret(
        name="scorer_api_key",
        value="scorer_secret",
        scope=SecretScope.SCORER.value,
        scope_id=123,
    )

    value = store.get_secret(name="scorer_api_key", scope=SecretScope.SCORER.value, scope_id=123)
    assert value == "scorer_secret"

    names = store.list_secret_names(scope=SecretScope.SCORER.value, scope_id=123)
    assert "scorer_api_key" in names

    with pytest.raises(MlflowException, match="not found"):
        store.get_secret(name="scorer_api_key", scope=SecretScope.SCORER.value, scope_id=999)


def test_corrupt_encrypted_data_fails_integrity_check():
    secret_manager = SecretManager()
    dek = secret_manager.generate_dek()
    encrypted_name = secret_manager.encrypt_with_dek("test_name", dek)
    encrypted_value = secret_manager.encrypt_with_dek("test_value", dek)
    encrypted_dek = secret_manager.encrypt_dek(dek)

    correct_hash = hashlib.sha256(b"test_name:test_value").hexdigest()

    corrupted_value = encrypted_value[:-10] + "corrupted!"

    from mlflow.server import app

    with app.test_client() as client:
        with mock.patch("mlflow.server.handlers._get_tracking_store") as mock_get_store:
            mock_store = mock.MagicMock()
            mock_get_store.return_value = mock_store

            response = client.post(
                "/api/3.0/mlflow/secrets/create",
                json={
                    "encrypted_name": encrypted_name,
                    "encrypted_value": corrupted_value,
                    "encrypted_dek": encrypted_dek,
                    "scope": SecretScope.GLOBAL.value,
                    "integrity_hash": correct_hash,
                },
            )

            assert response.status_code in [400, 500]


def test_client_computes_correct_integrity_hash():
    client = SecretsClient()

    name = "test_key"
    value = "test_value"

    expected_hash = hashlib.sha256(f"{name}:{value}".encode()).hexdigest()
    computed_hash = client._compute_integrity_hash(name, value)

    assert computed_hash == expected_hash

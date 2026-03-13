from pathlib import Path

import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.tracking.rest_store import RestStore
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.tracking.utils.secrets import get_decrypted_secret

TEST_PASSPHRASE = "test-passphrase"


@pytest.fixture(autouse=True)
def set_kek_passphrase(monkeypatch):
    monkeypatch.setenv("MLFLOW_CRYPTO_KEK_PASSPHRASE", TEST_PASSPHRASE)


@pytest.fixture
def store(tmp_path: Path):
    db_uri = f"sqlite:///{tmp_path}/test.db"
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir(exist_ok=True)
    return SqlAlchemyStore(db_uri, artifact_uri.as_uri())


def test_get_decrypted_secret_with_non_sqlalchemy_store():
    rest_store = RestStore(lambda: None)

    with pytest.raises(
        MlflowException,
        match="Secret retrieval is only supported with SqlAlchemyStore backends",
    ):
        get_decrypted_secret("test-secret-id", store=rest_store)


def test_get_decrypted_secret_integration_simple(store):
    secret_info = store.create_gateway_secret(
        secret_name="test-simple-secret",
        secret_value={"api_key": "sk-test-123456"},
        provider="openai",
    )

    decrypted = get_decrypted_secret(secret_info.secret_id, store=store)

    assert decrypted == {"api_key": "sk-test-123456"}


def test_get_decrypted_secret_integration_compound(store):
    secret_info = store.create_gateway_secret(
        secret_name="test-compound-secret",
        secret_value={
            "aws_access_key_id": "AKIA1234567890",
            "aws_secret_access_key": "secret-key-value",
        },
        provider="bedrock",
    )

    decrypted = get_decrypted_secret(secret_info.secret_id, store=store)

    assert decrypted == {
        "aws_access_key_id": "AKIA1234567890",
        "aws_secret_access_key": "secret-key-value",
    }


def test_get_decrypted_secret_integration_with_auth_config(store):
    secret_info = store.create_gateway_secret(
        secret_name="test-auth-config-secret",
        secret_value={"api_key": "aws-secret"},
        provider="bedrock",
        auth_config={"region": "us-east-1", "profile": "default"},
    )

    decrypted = get_decrypted_secret(secret_info.secret_id, store=store)

    assert decrypted == {"api_key": "aws-secret"}


def test_get_decrypted_secret_integration_not_found(store):
    with pytest.raises(MlflowException, match="not found"):
        get_decrypted_secret("nonexistent-secret-id", store=store)


def test_get_decrypted_secret_integration_multiple_secrets(store):
    secret1 = store.create_gateway_secret(
        secret_name="secret-1",
        secret_value={"api_key": "key-1"},
        provider="openai",
    )
    secret2 = store.create_gateway_secret(
        secret_name="secret-2",
        secret_value={"api_key": "key-2"},
        provider="anthropic",
    )

    decrypted1 = get_decrypted_secret(secret1.secret_id, store=store)
    decrypted2 = get_decrypted_secret(secret2.secret_id, store=store)

    assert decrypted1 == {"api_key": "key-1"}
    assert decrypted2 == {"api_key": "key-2"}
